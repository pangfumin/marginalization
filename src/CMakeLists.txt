# Build the main lib
add_library( ${PROJECT_NAME}
     factor/pose_local_parameterization.cpp
     factor/marginalization_factor.cpp
     factor/pinhole_project_factor.cpp
     factor/project_error.cpp
     factor/binary_pose_error_factor.cpp
     utility/utility.cpp
    )

# Build main lib
target_link_libraries( ${PROJECT_NAME}
        ${CERES_LIBRARIES})







