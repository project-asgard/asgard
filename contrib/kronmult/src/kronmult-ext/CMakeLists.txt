    cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
    
    if (USE_GPU)
        project(kronmult LANGUAGES CXX CUDA) 
        add_compile_definitions(USE_GPU)
    else ()
        project(kronmult LANGUAGES CXX) 
    endif ()
    
    add_compile_options ("-O3" "-g")
    
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        if(USE_MKL)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        endif()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else ()
	message("-- WARNING - building kronmult without openMP support ---")
    endif ()

    include(CTest)
    
    set(KRONSRC 
	kgemm_nn_batched.cpp 
	kgemm_nn_batched.hpp 
	kgemm_nn.cpp 
	kgemm_nn.hpp 
	kgemm_nt_batched.cpp 
	kgemm_nt_batched.hpp 
	kgemm_nt.cpp 
	kgemm_nt.hpp 
	kroncommon.hpp 
	kronmult1_batched.cpp 
	kronmult1_batched.hpp 
	kronmult1_pbatched.cpp 
	kronmult1_pbatched.hpp 
	kronmult1_xbatched.cpp
	kronmult1_xbatched.hpp
	kronmult1.hpp 
	kronmult2_batched.cpp 
	kronmult2_batched.hpp 
	kronmult2_pbatched.cpp 
	kronmult2_pbatched.hpp
	kronmult2_xbatched.cpp
	kronmult2_xbatched.hpp
	kronmult2.hpp 
	kronmult3_batched.cpp 
	kronmult3_batched.hpp 
	kronmult3_pbatched.cpp 
	kronmult3_pbatched.hpp
	kronmult3_xbatched.cpp
	kronmult3_xbatched.hpp
	kronmult3.hpp 
	kronmult4_batched.cpp 
	kronmult4_batched.hpp 
	kronmult4_pbatched.cpp 
	kronmult4_pbatched.hpp 
	kronmult4_xbatched.cpp
	kronmult4_xbatched.hpp
	kronmult4.hpp 
	kronmult5_batched.cpp 
	kronmult5_batched.hpp 
	kronmult5_pbatched.cpp 
	kronmult5_pbatched.hpp 
	kronmult5_xbatched.cpp
	kronmult5_xbatched.hpp
	kronmult5.hpp 
	kronmult6_batched.cpp 
	kronmult6_batched.hpp 
	kronmult6_pbatched.hpp 
	kronmult6_pbatched.cpp 
	kronmult6_xbatched.cpp
	kronmult6_xbatched.hpp
	kronmult6.hpp 
      )
   
    if(USE_GPU)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch sm_60 --compiler-options -fPIC --keep-device-functions")
        if(USE_MKL)
	  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch sm_60 --compiler-options -fopenmp -fPIC --keep-device-functions")
        endif()
	set_source_files_properties( ${KRONSRC} PROPERTIES LANGUAGE CUDA )
    endif()
  

    add_library(kron SHARED ${KRONSRC})
    target_compile_features(kron PUBLIC cxx_std_11)
   
    set(TESTSRC 
	test_kronmult6_batched.cpp 
	test_kronmult6_pbatched.cpp
        test_kronmult6_xbatched.cpp	
	test_kgemm_nn_batched.cpp 
	test_kgemm_nt_batched.cpp 
      )

    if(USE_GPU)
	    set_source_files_properties( ${TESTSRC} PROPERTIES LANGUAGE CUDA )
    endif ()


    add_executable(kgemm_nn_test test_kgemm_nn_batched.cpp) 
    add_executable(kgemm_nt_test test_kgemm_nt_batched.cpp)
    add_executable(test_kronmult6_batched test_kronmult6_batched.cpp)
    add_executable(test_kronmult6_pbatched test_kronmult6_pbatched.cpp) 
    add_executable(test_kronmult6_xbatched test_kronmult6_xbatched.cpp)
    
    target_link_libraries(kgemm_nn_test PRIVATE kron) 
    target_link_libraries(kgemm_nt_test PRIVATE kron)
    target_link_libraries(test_kronmult6_batched PRIVATE kron)
    target_link_libraries(test_kronmult6_pbatched PRIVATE kron) 
    target_link_libraries(test_kronmult6_xbatched PRIVATE kron)
