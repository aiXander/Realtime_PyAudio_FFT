import numpy as np
import math, scipy, pygame

def round_up_to_even(f):
    return int(math.ceil(f / 2.) * 2)

def round_to_nearest_power_of_two(f, base=2):
    l = math.log(f,base)
    rounded = int(np.round(l,0))
    return base**rounded

def get_frequency_bins(start, stop, n):
    octaves = np.logspace(log(start)/log(2), log(stop)/log(2), n, endpoint=True, base=2, dtype=None)
    return np.insert(octaves, 0, 0)

def gaussian_kernel1d(sigma, truncate=2.0):
    sigma = float(sigma)
    sigma2 = sigma * sigma
    # make the radius of the filter equal to truncate standard deviations
    radius = int(truncate * sigma + 0.5)
    exponent_range = np.arange(1)
    
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    return phi_x

def gaussian_kernel_1D(w, sigma):
    sigma = sigma
    x = np.linspace(-sigma, sigma, w+1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    return kern1d/kern1d.sum()

def get_smoothing_filter(FFT_window_size_ms, filter_length_ms, verbose = 0):
    buffer_length = round_up_to_even(filter_length_ms / FFT_window_size_ms)+1
    filter_sigma = buffer_length / 3  #How quickly the smoothing influence drops over the buffer length
    filter_weights = gaussian_kernel1d(filter_sigma)[:,np.newaxis]

    max_index = np.argmax(filter_weights)
    filter_weights = filter_weights[:max_index+1]
    filter_weights = filter_weights / np.mean(filter_weights)

    if verbose:
        min_fraction = 100*np.min(filter_weights)/np.max(filter_weights)
        print('\nApplying temporal smoothing to the FFT features...')
        print("Smoothing buffer contains %d FFT windows (sigma: %.3f) --> min_contribution: %.3f%%" %(buffer_length, filter_sigma, min_fraction))
        print("Filter weights:")
        for i, w in enumerate(filter_weights):
            print("%02d: %.3f" %(len(filter_weights)-i, w))

    return filter_weights

class numpy_data_buffer:
    """
    A fast, circular FIFO buffer in numpy with minimal memory interactions by using an array of index pointers
    """

    def __init__(self, n_windows, samples_per_window, dtype = np.float32, start_value = 0, data_dimensions = 1):
        self.n_windows = n_windows
        self.data_dimensions = data_dimensions
        self.samples_per_window = samples_per_window
        self.data = start_value * np.ones((self.n_windows, self.samples_per_window), dtype = dtype)

        if self.data_dimensions == 1:
            self.total_samples = self.n_windows * self.samples_per_window
        else:
            self.total_samples = self.n_windows

        self.elements_in_buffer = 0
        self.overwrite_index = 0

        self.indices = np.arange(self.n_windows, dtype=np.int32)
        self.last_window_id = np.max(self.indices)
        self.index_order = np.argsort(self.indices)

    def append_data(self, data_window):
        self.data[self.overwrite_index, :] = data_window

        self.last_window_id += 1
        self.indices[self.overwrite_index] = self.last_window_id
        self.index_order = np.argsort(self.indices)

        self.overwrite_index += 1
        self.overwrite_index = self.overwrite_index % self.n_windows

        self.elements_in_buffer += 1
        self.elements_in_buffer = min(self.n_windows, self.elements_in_buffer)

    def get_most_recent(self, window_size):
        ordered_dataframe = self.data[self.index_order]
        if self.data_dimensions == 1:
            ordered_dataframe = np.hstack(ordered_dataframe)
        return ordered_dataframe[self.total_samples - window_size:]

    def get_buffer_data(self):
        return self.data[:self.elements_in_buffer]

class Button:
    def __init__(self, text="", right=10, top=30, width=None, height=20):
        self.text = text
        self.top = top
        self.height = height
        self.colour1 = (220, 220, 220)  # main
        self.colour2 = (100, 100, 100)  # border
        self.colour3 = (172, 220, 247)  # hover
        self.colour4 = (225, 243, 252)
        self.fontname = "freesansbold.ttf"
        self.fontsize = self.height-6
        self.mouse_over = False
        self.mouse_down = False
        self.mouse = "off"
        self.clicked = False
        self.pyg = pygame
        self.font = pygame.font.SysFont(self.fontname, self.fontsize)
        self.text_width, self.text_height = self.pyg.font.Font.size(self.font, self.text)
        if width == None:
            self.width = int(self.text_width * 1.3)
            self.width_type = "text"
        else:
            self.width = width
            self.width_type = "user"

        self.left = right - self.width
        self.buttonUP = self.pyg.Surface((self.width, self.height))
        self.buttonDOWN = self.pyg.Surface((self.width, self.height))
        self.buttonHOVER = self.pyg.Surface((self.width, self.height))
        self.__update__()

    def __update__(self):
        # up
        r, g, b = self.colour1
        self.buttonUP.fill(self.colour1)
        self.pyg.draw.rect(self.buttonUP, (r+20, g+20, b+20), (0, 0, self.width, self.height/2), 0)
        self.pyg.draw.line(self.buttonUP, self.colour2, (2, 0), (self.width-3, 0), 1)
        self.pyg.draw.line(self.buttonUP, self.colour2, (2, self.height-1), (self.width-3, self.height-1), 1)
        self.pyg.draw.line(self.buttonUP, self.colour2, (0, 2), (0, self.height-3), 1)
        self.pyg.draw.line(self.buttonUP, self.colour2, (self.width-1, 2), (self.width-1, self.height-3), 1)
        self.buttonUP.set_at((1, 1), self.colour2)
        self.buttonUP.set_at((self.width-2, 1), self.colour2)
        self.buttonUP.set_at((1, self.height-2), self.colour2)
        self.buttonUP.set_at((self.width-2, self.height-2), self.colour2)
        self.buttonUP.blit(self.font.render(self.text, False, (0, 0, 0)), ((self.width/2)-(self.text_width/2), (self.height/2)-(self.text_height/2)))
        # hover
        self.buttonHOVER.fill(self.colour3)
        self.pyg.draw.rect(self.buttonHOVER, self.colour4, (0, 0, self.width, self.height/2), 0)
        self.pyg.draw.line(self.buttonHOVER, self.colour2, (2, 0), (self.width-3, 0), 1)
        self.pyg.draw.line(self.buttonHOVER, self.colour2, (2, self.height-1), (self.width-3, self.height-1), 1)
        self.pyg.draw.line(self.buttonHOVER, self.colour4, (2, self.height-2), (self.width-3, self.height-2), 1)
        self.pyg.draw.line(self.buttonHOVER, self.colour2, (0, 2), (0, self.height-3), 1)
        self.pyg.draw.line(self.buttonHOVER, self.colour4, (1, 2), (1, self.height-3), 2)
        self.pyg.draw.line(self.buttonHOVER, self.colour2, (self.width-1, 2), (self.width-1, self.height-3), 1)
        self.buttonHOVER.set_at((1, 1), self.colour2)
        self.buttonHOVER.set_at((self.width-2, 1), self.colour2)
        self.buttonHOVER.set_at((1, self.height-2), self.colour2)
        self.buttonHOVER.set_at((self.width-2, self.height-2), self.colour2)
        self.buttonHOVER.blit(self.font.render(self.text, False, (0, 0, 0)), ((self.width/2)-(self.text_width/2), (self.height/2)-(self.text_height/2)))
        # down
        r, g, b = self.colour3
        r2, g2, b2 = self.colour4
        self.buttonDOWN.fill((r-20, g-20, b-10))
        self.pyg.draw.rect(self.buttonDOWN, (r2-20, g2-20, b2-10), (0, 0, self.width, self.height/2), 0)
        self.pyg.draw.line(self.buttonDOWN, self.colour2, (2, 0), (self.width-3, 0), 1)
        self.pyg.draw.line(self.buttonDOWN, (r-20, g-20, b-10), (2, 1), (self.width-3, 1), 2)
        self.pyg.draw.line(self.buttonDOWN, self.colour2, (2, self.height-1), (self.width-3, self.height-1), 1)
        self.pyg.draw.line(self.buttonDOWN, self.colour2, (0, 2), (0, self.height-3), 1)
        self.pyg.draw.line(self.buttonDOWN, (r-20, g-20, b-10), (1, 2), (1, self.height-3), 2)
        self.pyg.draw.line(self.buttonDOWN, self.colour2, (self.width-1, 2), (self.width-1, self.height-3), 1)
        self.buttonDOWN.set_at((1, 1), self.colour2)
        self.buttonDOWN.set_at((self.width-2, 1), self.colour2)
        self.buttonDOWN.set_at((1, self.height-2), self.colour2)
        self.buttonDOWN.set_at((self.width-2, self.height-2), self.colour2)
        self.buttonDOWN.blit(self.font.render(self.text, False, (0, 0, 0)), ((self.width/2)-(self.text_width/2)+1, (self.height/2)-(self.text_height/2)))

    def draw(self, surface):
        self.__mouse_check__()
        if self.mouse == "hover":
            surface.blit(self.buttonHOVER, (self.left, self.top))
        elif self.mouse == "off":
            surface.blit(self.buttonUP, (self.left, self.top))
        elif self.mouse == "down":
            surface.blit(self.buttonDOWN, (self.left, self.top))

    def __mouse_check__(self):
        _1, _2, _3 = pygame.mouse.get_pressed()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if not _1:
            self.mouse = "off"
        if mouse_x > self.left and mouse_x < self.left + self.width and mouse_y > self.top and mouse_y < self.top + self.height and not self.mouse == "down":
            self.mouse = "hover"
        if not self.mouse_down and _1 and self.mouse == "hover":
            self.mouse = "down"
            self.clicked = True
        if self.mouse == "off":
            self.clicked = False

    def click(self):
        _1, _2, _3 = pygame.mouse.get_pressed()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if mouse_x > self.left and mouse_x < self.left + self.width and mouse_y > self.top and mouse_y < self.top + self.height and self.clicked and not _1:
            self.clicked = False
            return True
        else:
            return False

    def set_text(self, text, fontname="Arial", fontsize=None):
        self.text = text
        self.fontname = fontname
        if not fontsize == None:
            self.fontsize = fontsize
        self.font = pygame.font.SysFont(self.fontname, self.fontsize)
        self.text_width, self.text_height = self.pyg.font.Font.size(self.font, self.text)
        if self.width_type == "text":
            self.width = self.text_width + 20
        self.buttonUP = self.pyg.Surface((self.width, self.height))
        self.buttonDOWN = self.pyg.Surface((self.width, self.height))
        self.buttonHOVER = self.pyg.Surface((self.width, self.height))
        self.__update__()