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
include performance/CMakeFiles/coalescing.dir/depend.make

# Include the progress variables for this target.
include performance/CMakeFiles/coalescing.dir/progress.make

# Include the compile flags for this target's objects.
include performance/CMakeFiles/coalescing.dir/flags.make

performance/CMakeFiles/coalescing.dir/coalescing_generated_02-coalescing.cu.o: performance/CMakeFiles/coalescing.dir/coalescing_generated_02-coalescing.cu.o.depend
performance/CMakeFiles/coalescing.dir/coalescing_generated_02-coalescing.cu.o: performance/CMakeFiles/coalescing.dir/coalescing_generated_02-coalescing.cu.o.cmake
performance/CMakeFiles/coalescing.dir/coalescing_generated_02-coalescing.cu.o: performance/02-coalescing.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/scratch/e802756/COURSES/SSPP/2022-2023/solutions/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object performance/CMakeFiles/coalescing.dir/coalescing_generated_02-coalescing.cu.o"
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/coalescing.dir && /apps/software/CMake/3.15.3-GCCcore-8.3.0/bin/cmake -E make_directory /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/coalescing.dir//.
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/coalescing.dir && /apps/software/CMake/3.15.3-GCCcore-8.3.0/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/coalescing.dir//./coalescing_generated_02-coalescing.cu.o -D generated_cubin_file:STRING=/scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/coalescing.dir//./coalescing_generated_02-coalescing.cu.o.cubin.txt -P /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/coalescing.dir//coalescing_generated_02-coalescing.cu.o.cmake

# Object files for target coalescing
coalescing_OBJECTS =

# External object files for target coalescing
coalescing_EXTERNAL_OBJECTS = \
"/scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/coalescing.dir/coalescing_generated_02-coalescing.cu.o"

performance/coalescing: performance/CMakeFiles/coalescing.dir/coalescing_generated_02-coalescing.cu.o
performance/coalescing: performance/CMakeFiles/coalescing.dir/build.make
performance/coalescing: /apps/software/CUDA/10.1.243-GCC-8.3.0/lib64/libcudart_static.a
performance/coalescing: /usr/lib64/librt.so
performance/coalescing: performance/CMakeFiles/coalescing.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/scratch/e802756/COURSES/SSPP/2022-2023/solutions/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable coalescing"
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/coalescing.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
performance/CMakeFiles/coalescing.dir/build: performance/coalescing

.PHONY : performance/CMakeFiles/coalescing.dir/build

performance/CMakeFiles/coalescing.dir/clean:
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance && $(CMAKE_COMMAND) -P CMakeFiles/coalescing.dir/cmake_clean.cmake
.PHONY : performance/CMakeFiles/coalescing.dir/clean

performance/CMakeFiles/coalescing.dir/depend: performance/CMakeFiles/coalescing.dir/coalescing_generated_02-coalescing.cu.o
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /scratch/e802756/COURSES/SSPP/2022-2023/solutions /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance /scratch/e802756/COURSES/SSPP/2022-2023/solutions /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/coalescing.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : performance/CMakeFiles/coalescing.dir/depend
