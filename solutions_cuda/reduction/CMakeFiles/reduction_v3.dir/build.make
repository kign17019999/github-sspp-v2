# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /apps/software/CMake/3.15.3-GCCcore-8.3.0/bin/cmake

# The command to remove a file.
RM = /apps/software/CMake/3.15.3-GCCcore-8.3.0/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /scratch/e802756/COURSES/SSPP/2022-2023/solutions

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /scratch/e802756/COURSES/SSPP/2022-2023/solutions

# Include any dependencies generated for this target.
include reduction/CMakeFiles/reduction_v3.dir/depend.make

# Include the progress variables for this target.
include reduction/CMakeFiles/reduction_v3.dir/progress.make

# Include the compile flags for this target's objects.
include reduction/CMakeFiles/reduction_v3.dir/flags.make

reduction/CMakeFiles/reduction_v3.dir/reduction_v3_generated_reduction_v3.cu.o: reduction/CMakeFiles/reduction_v3.dir/reduction_v3_generated_reduction_v3.cu.o.depend
reduction/CMakeFiles/reduction_v3.dir/reduction_v3_generated_reduction_v3.cu.o: reduction/CMakeFiles/reduction_v3.dir/reduction_v3_generated_reduction_v3.cu.o.cmake
reduction/CMakeFiles/reduction_v3.dir/reduction_v3_generated_reduction_v3.cu.o: reduction/reduction_v3.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/scratch/e802756/COURSES/SSPP/2022-2023/solutions/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object reduction/CMakeFiles/reduction_v3.dir/reduction_v3_generated_reduction_v3.cu.o"
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions/reduction/CMakeFiles/reduction_v3.dir && /apps/software/CMake/3.15.3-GCCcore-8.3.0/bin/cmake -E make_directory /scratch/e802756/COURSES/SSPP/2022-2023/solutions/reduction/CMakeFiles/reduction_v3.dir//.
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions/reduction/CMakeFiles/reduction_v3.dir && /apps/software/CMake/3.15.3-GCCcore-8.3.0/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/scratch/e802756/COURSES/SSPP/2022-2023/solutions/reduction/CMakeFiles/reduction_v3.dir//./reduction_v3_generated_reduction_v3.cu.o -D generated_cubin_file:STRING=/scratch/e802756/COURSES/SSPP/2022-2023/solutions/reduction/CMakeFiles/reduction_v3.dir//./reduction_v3_generated_reduction_v3.cu.o.cubin.txt -P /scratch/e802756/COURSES/SSPP/2022-2023/solutions/reduction/CMakeFiles/reduction_v3.dir//reduction_v3_generated_reduction_v3.cu.o.cmake

# Object files for target reduction_v3
reduction_v3_OBJECTS =

# External object files for target reduction_v3
reduction_v3_EXTERNAL_OBJECTS = \
"/scratch/e802756/COURSES/SSPP/2022-2023/solutions/reduction/CMakeFiles/reduction_v3.dir/reduction_v3_generated_reduction_v3.cu.o"

reduction/reduction_v3: reduction/CMakeFiles/reduction_v3.dir/reduction_v3_generated_reduction_v3.cu.o
reduction/reduction_v3: reduction/CMakeFiles/reduction_v3.dir/build.make
reduction/reduction_v3: /apps/software/CUDA/10.1.243-GCC-8.3.0/lib64/libcudart_static.a
reduction/reduction_v3: /usr/lib64/librt.so
reduction/reduction_v3: reduction/CMakeFiles/reduction_v3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/scratch/e802756/COURSES/SSPP/2022-2023/solutions/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable reduction_v3"
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions/reduction && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reduction_v3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
reduction/CMakeFiles/reduction_v3.dir/build: reduction/reduction_v3

.PHONY : reduction/CMakeFiles/reduction_v3.dir/build

reduction/CMakeFiles/reduction_v3.dir/clean:
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions/reduction && $(CMAKE_COMMAND) -P CMakeFiles/reduction_v3.dir/cmake_clean.cmake
.PHONY : reduction/CMakeFiles/reduction_v3.dir/clean

reduction/CMakeFiles/reduction_v3.dir/depend: reduction/CMakeFiles/reduction_v3.dir/reduction_v3_generated_reduction_v3.cu.o
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /scratch/e802756/COURSES/SSPP/2022-2023/solutions /scratch/e802756/COURSES/SSPP/2022-2023/solutions/reduction /scratch/e802756/COURSES/SSPP/2022-2023/solutions /scratch/e802756/COURSES/SSPP/2022-2023/solutions/reduction /scratch/e802756/COURSES/SSPP/2022-2023/solutions/reduction/CMakeFiles/reduction_v3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : reduction/CMakeFiles/reduction_v3.dir/depend

