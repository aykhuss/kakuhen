add_executable(example example.cpp)
target_link_libraries(example PRIVATE kakuhen)

add_executable(example_functor example_functor.cpp)
target_link_libraries(example_functor PRIVATE kakuhen)

add_executable(playground playground.cpp)
target_link_libraries(playground PRIVATE kakuhen)

add_executable(plot_data plot_data.cpp)
target_link_libraries(plot_data PRIVATE kakuhen)
