set(EXAMPLES
  example
  example_functor
  example_histogram
  example_generator
  example_generator_benchmark
  example_generate_n_events
  example_benchmark
  example_live_progress
  # example_mapping
  # plot_data
  # playground
)

foreach(example ${EXAMPLES})
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} PRIVATE kakuhen kakuhen_warnings)
endforeach()
