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
include performance/CMakeFiles/memory_transfers.dir/depend.make

# Include the progress variables for this target.
include performance/CMakeFiles/memory_transfers.dir/progress.make

# Include the compile flags for this target's objects.
include performance/CMakeFiles/memory_transfers.dir/flags.make

performance/CMakeFiles/memory_transfers.dir/memory_transfers_generated_01-memory_transfers.cu.o: performance/CMakeFiles/memory_transfers.dir/memory_transfers_generated_01-memory_transfers.cu.o.depend
performance/CMakeFiles/memory_transfers.dir/memory_transfers_generated_01-memory_transfers.cu.o: performance/CMakeFiles/memory_transfers.dir/memory_transfers_generated_01-memory_transfers.cu.o.cmake
performance/CMakeFiles/memory_transfers.dir/memory_transfers_generated_01-memory_transfers.cu.o: performance/01-memory_transfers.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/scratch/e802756/COURSES/SSPP/2022-2023/solutions/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object performance/CMakeFiles/memory_transfers.dir/memory_transfers_generated_01-memory_transfers.cu.o"
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/memory_transfers.dir && /apps/software/CMake/3.15.3-GCCcore-8.3.0/bin/cmake -E make_directory /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/memory_transfers.dir//.
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/memory_transfers.dir && /apps/software/CMake/3.15.3-GCCcore-8.3.0/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/memory_transfers.dir//./memory_transfers_generated_01-memory_transfers.cu.o -D generated_cubin_file:STRING=/scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/memory_transfers.dir//./memory_transfers_generated_01-memory_transfers.cu.o.cubin.txt -P /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/memory_transfers.dir//memory_transfers_generated_01-memory_transfers.cu.o.cmake

# Object files for target memory_transfers
memory_transfers_OBJECTS =

# External object files for target memory_transfers
memory_transfers_EXTERNAL_OBJECTS = \
"/scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/memory_transfers.dir/memory_transfers_generated_01-memory_transfers.cu.o"

performance/memory_transfers: performance/CMakeFiles/memory_transfers.dir/memory_transfers_generated_01-memory_transfers.cu.o
performance/memory_transfers: performance/CMakeFiles/memory_transfers.dir/build.make
performance/memory_transfers: /apps/software/CUDA/10.1.243-GCC-8.3.0/lib64/libcudart_static.a
performance/memory_transfers: /usr/lib64/librt.so
performance/memory_transfers: performance/CMakeFiles/memory_transfers.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/scratch/e802756/COURSES/SSPP/2022-2023/solutions/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable memory_transfers"
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/memory_transfers.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
performance/CMakeFiles/memory_transfers.dir/build: performance/memory_transfers

.PHONY : performance/CMakeFiles/memory_transfers.dir/build

performance/CMakeFiles/memory_transfers.dir/clean:
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance && $(CMAKE_COMMAND) -P CMakeFiles/memory_transfers.dir/cmake_clean.cmake
.PHONY : performance/CMakeFiles/memory_transfers.dir/clean

performance/CMakeFiles/memory_transfers.dir/depend: performance/CMakeFiles/memory_transfers.dir/memory_transfers_generated_01-memory_transfers.cu.o
	cd /scratch/e802756/COURSES/SSPP/2022-2023/solutions && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /scratch/e802756/COURSES/SSPP/2022-2023/solutions /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance /scratch/e802756/COURSES/SSPP/2022-2023/solutions /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance /scratch/e802756/COURSES/SSPP/2022-2023/solutions/performance/CMakeFiles/memory_transfers.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : performance/CMakeFiles/memory_transfers.dir/depend

