# Compiler and Flags
CXX = g++
CXXFLAGS = -shared -fPIC -std=c++17 -O2 -g
PYBIND11_CFLAGS := $(shell python3 -m pybind11 --includes)  # Fetch pybind11 flags
PYTHON_CFLAGS := $(shell python3-config --includes)
INCLUDES = -I./src/surge/src -I./src/evaluation $(PYBIND11_CFLAGS) $(PYTHON_CFLAGS) -I/usr/include/python3.10

# Default to CPU-only compilation
DEFAULT_LIBS =
DEFAULT_DEFINES =
DEFAULT_LDFLAGS =

# Source Files
SRCS := $(wildcard src/*.cpp src/surge/src/*.cpp src/evaluation/*.cpp)
OBJS := $(SRCS:.cpp=.o)

# Output Python Module
TARGET = wrapper.so

# Check if HAS_TORCH is defined in the header file
ifdef HAS_TORCH
LIBS = -ltorch -lc10 -ltorch_cpu
DEFINES = -DHAS_TORCH
LDFLAGS = -L./src/libtorch/lib -Wl,-rpath=./src/libtorch/lib
INCLUDES += -I./src/libtorch/include -I./src/libtorch/include/torch/csrc/api/include
else
LIBS = $(DEFAULT_LIBS)
DEFINES = $(DEFAULT_DEFINES)
LDFLAGS = $(DEFAULT_LDFLAGS)
endif

# Build the Python Extension Module
.PHONY: all cpu gpu
all: check_libtorch cpu  # Default to CPU-only

cpu: $(TARGET)

gpu: check_libtorch TORCH_GPU_FLAGS
	@$(MAKE) $(TARGET)

TORCH_GPU_FLAGS:
	@echo "Compiling with libtorch GPU support"
	$(eval LIBS := -ltorch -lc10 -ltorch_cpu -ltorch_gpu)
	$(eval DEFINES := -DHAS_TORCH)

check_libtorch:
ifdef HAS_TORCH
	@if [ ! -d "./src/libtorch" ]; then \
		echo "Error: ./src/libtorch directory does not exist. Ensure libtorch is installed or remove HAS_TORCH definition."; \
		exit 1; \
	fi
endif

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(DEFINES) $(OBJS) $(LDFLAGS) $(LIBS) -o $(TARGET)

# Compile source files into object files
src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) -c $< -o $@

src/surge/src/%.o: src/surge/src/%.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) -c $< -o $@

src/evaluation/%.o: src/evaluation/%.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) -c $< -o $@

# Clean only object files
.PHONY: clean_objs
clean_objs:
	rm -f $(OBJS)

# Clean everything
.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
