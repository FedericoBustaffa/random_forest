# compiler
CXX = g++

# general flags
CXXFLAGS = -Wall -std=c++20

# flags for debug compilation - enabled if run "make -DBUILD_TYPE=DEBUG"
DBGFLAGS = -g

# flags for optimized compilation - disabled if compiled in debug mode
OPTFLAGS = -O2 -march=native

# dependencies flags
DEPSFLAGS = -MMD -MP

# specify include directories with -I<dir>
INCLUDES = -I./include/ -I./csv-parser/single_include/

# specify preprocessor definitions
DEFINES = 

# convenient single variable to wrap all the flags
FLAGS = $(CXXFLAGS)
FLAGS += $(INCLUDES)
FLAGS += $(DEFINES)

# default mode is RELEASE
BUILD_TYPE ?= RELEASE
ifeq ($(BUILD_TYPE),DEBUG)
	FLAGS += $(DBGFLAGS)
else ifeq ($(BUILD_TYPE),RELEASE)
	FLAGS += $(OPTFLAGS)
else
	$(error "Invalid BUILD_TYPE. Use DEBUG or RELEASE.")
endif

# link libraries
LIBS =

# specify source directory
SOURCE_DIR = ./src
SOURCES = $(wildcard $(SOURCE_DIR)/*.cpp)
TESTS = $(wildcard test/*.cpp)

# build directory containing .o and .d files
BUILD_DIR = build

# directory for .d files
DEPS = $(patsubst $(SOURCE_DIR)/%.cpp, $(BUILD_DIR)/%.d, $(SOURCES))

# generate the object files based on the sources names
OBJECTS = $(patsubst $(SOURCE_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SOURCES))

.PHONY: all clean-fast clean recompile

TARGETS = $(patsubst test/%.cpp, test/%.out, $(TESTS))

all: $(BUILD_DIR) test/$(TARGETS)

$(BUILD_DIR):
	@mkdir -p $@

test/%.out: $(OBJECTS) test/%.cpp
	$(CXX) $(FLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp
	$(CXX) $(FLAGS) $(DEPSFLAGS) -c $< -o $@

-include $(DEPS)

clean-fast:
	-rm -rf $(BUILD_DIR)

clean: clean-fast
	-rm -rf $(TARGETS)

recompile: clean all
