# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zsq/Documents/VScode_file/CUDA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zsq/Documents/VScode_file/CUDA/build

# Include any dependencies generated for this target.
include CMakeFiles/cudaMatMul.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cudaMatMul.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cudaMatMul.dir/flags.make

CMakeFiles/cudaMatMul.dir/cudaMatMul.cu.o: CMakeFiles/cudaMatMul.dir/flags.make
CMakeFiles/cudaMatMul.dir/cudaMatMul.cu.o: ../cudaMatMul.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zsq/Documents/VScode_file/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/cudaMatMul.dir/cudaMatMul.cu.o"
	/usr/local/cuda-11.2/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/zsq/Documents/VScode_file/CUDA/cudaMatMul.cu -o CMakeFiles/cudaMatMul.dir/cudaMatMul.cu.o

CMakeFiles/cudaMatMul.dir/cudaMatMul.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cudaMatMul.dir/cudaMatMul.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cudaMatMul.dir/cudaMatMul.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cudaMatMul.dir/cudaMatMul.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cudaMatMul
cudaMatMul_OBJECTS = \
"CMakeFiles/cudaMatMul.dir/cudaMatMul.cu.o"

# External object files for target cudaMatMul
cudaMatMul_EXTERNAL_OBJECTS =

libcudaMatMul.a: CMakeFiles/cudaMatMul.dir/cudaMatMul.cu.o
libcudaMatMul.a: CMakeFiles/cudaMatMul.dir/build.make
libcudaMatMul.a: CMakeFiles/cudaMatMul.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zsq/Documents/VScode_file/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA static library libcudaMatMul.a"
	$(CMAKE_COMMAND) -P CMakeFiles/cudaMatMul.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cudaMatMul.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cudaMatMul.dir/build: libcudaMatMul.a

.PHONY : CMakeFiles/cudaMatMul.dir/build

CMakeFiles/cudaMatMul.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cudaMatMul.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cudaMatMul.dir/clean

CMakeFiles/cudaMatMul.dir/depend:
	cd /home/zsq/Documents/VScode_file/CUDA/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zsq/Documents/VScode_file/CUDA /home/zsq/Documents/VScode_file/CUDA /home/zsq/Documents/VScode_file/CUDA/build /home/zsq/Documents/VScode_file/CUDA/build /home/zsq/Documents/VScode_file/CUDA/build/CMakeFiles/cudaMatMul.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cudaMatMul.dir/depend
