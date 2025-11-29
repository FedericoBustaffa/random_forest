# compiler
CXX = g++

# general flags
CXXFLAGS = -Wall -std=c++20 -fopenmp

# flags for debug compilation - enabled if run "make -DBUILD_TYPE=DEBUG"
DBGFLAGS = -g -DDEBUG

# flags for optimized compilation - disabled if compiled in debug mode
OPTFLAGS = -O3 -march=native -DNDEBUG

# dependencies flags
DEPSFLAGS = -MMD -MP

# include directories
INCLUDES = -I./include/ -I./fastflow/

# specify preprocessor definitions
DEFINES = 

# convenient single variable to wrap all the flags
FLAGS = $(CXXFLAGS) $(INCLUDES) $(DEFINES)

# link libraries
LIBS = -pthread

# specify source directory
SOURCES = $(wildcard src/*.cpp)
TEST_SOURCES = $(wildcard test/*.cpp)

# build directory containing .o and .d files
BUILD_DIR = build

# directory for .d files
DEPS = $(patsubst src/%.cpp, $(BUILD_DIR)/%.d, $(SOURCES))
DEPS += $(patsubst test/%.cpp, $(BUILD_DIR)/test_%.d, $(TEST_SOURCES))

# generate the object files based on the sources names
OBJECTS = $(patsubst src/%.cpp, $(BUILD_DIR)/%.o, $(SOURCES))
TEST_OBJECTS = $(patsubst test/%.cpp, $(BUILD_DIR)/test_%.o, $(TEST_SOURCES))

# generate targets executables
TARGETS = $(patsubst test/%.cpp, $(BUILD_DIR)/%.out, $(TEST_SOURCES))

.PHONY: all clean-fast clean recompile


all: FLAGS += $(OPTFLAGS)
all: $(BUILD_DIR) $(TARGETS)

debug: FLAGS += $(DBGFLAGS)
debug: $(BUILD_DIR) $(TARGETS)


$(BUILD_DIR):
	@mkdir -p $@

$(BUILD_DIR)/%.o: src/%.cpp
	$(CXX) $(FLAGS) $(DEPSFLAGS) -c $< -o $@

$(BUILD_DIR)/test_%.o: test/%.cpp
	$(CXX) $(FLAGS) $(DEPSFLAGS) -c $< -o $@

$(BUILD_DIR)/%.out: $(OBJECTS) $(BUILD_DIR)/test_%.o
	$(CXX) $(FLAGS) $^ -o $@

-include $(DEPS)

clean-fast:
	-rm -rf $(BUILD_DIR)

clean: clean-fast
	-rm -rf $(TARGETS) ./test/*.out

