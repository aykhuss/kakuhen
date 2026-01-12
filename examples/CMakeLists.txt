set(EXAMPLES
  example
  example_functor
  # example_mapping
  # plot_data
  # playground
)

foreach(example ${EXAMPLES})
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} PRIVATE kakuhen kakuhen_warnings)
endforeach()
