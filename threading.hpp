#include <fmt/format.h>
#include <fmt/compile.h>
#include <fmt/color.h>

#include <thread>
#include <functional>
#include <semaphore>


// Provides general threading infrastructure and helper functions
//
// The functions provided are do_threaded_without_pool, do_threaded, and loop_threaded
// Constant thread_count is also provided
//
// The thread_setup namespace should not be used elsewhere


const std::size_t thread_count{std::max(1u, std::thread::hardware_concurrency() - 1)};
//constexpr std::size_t thread_count{4};

// create a thread for each core and run the given function on each
// the function must take 0 arguments, or a single std::uint32_t,
// in which case the thread number [0, thread_count - 1]  will be passed
void do_threaded_without_pool(auto func)
{
  std::vector<std::jthread> loop_threads{};

  loop_threads.reserve(thread_count);

  for (std::uint32_t thread_index = 0; thread_index < thread_count; ++thread_index)
    if constexpr (std::invocable<decltype(func)>)
      loop_threads.emplace_back(func);
    else
    {
      static_assert(std::invocable<decltype(func), std::uint32_t>, "test_repeat must be passed a function callable with 0 or 1 arguments");

      loop_threads.emplace_back(func, thread_index);
    }
}


namespace thread_setup
{
std::vector<std::function<void()>> thread_functions(thread_count);

std::array thread_blockers =
  ( []<std::size_t... I>(std::index_sequence<I...>)
    { return std::array<std::binary_semaphore, sizeof...(I)>{{ ((void) I, std::binary_semaphore(0))... }}; }
  )(std::make_index_sequence<192>{});

std::array thread_completers =
  ( []<std::size_t... I>(std::index_sequence<I...>)
    { return std::array<std::binary_semaphore, sizeof...(I)>{{ ((void) I, std::binary_semaphore(0))... }}; }
  )(std::make_index_sequence<192>{});

// Set to true when wanting to exit the program
bool terminate_threads{false};

// Run on every worker thread
void thread_loop(const std::uint32_t thread_index)
{
  while (true)
  {
    thread_blockers[thread_index].acquire();

    thread_functions[thread_index]();

    thread_completers[thread_index].release();

    if (terminate_threads)
      return;
  }
}

// Create the worker threads at startup
auto threads = []
{
  if (thread_count > thread_blockers.size())
  {
    fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "Attempting to create {} threads, but the limit is {}.\n", thread_count, thread_blockers.size());

    std::exit(EXIT_FAILURE);
  }

  std::vector<std::jthread> out;

  out.reserve(thread_count);

  for (std::size_t thread_index = 0; thread_index < thread_count; ++thread_index)
  {
    out.emplace_back(thread_loop, thread_index);

    out[thread_index].detach();
  }

  return out;
}();
} // namespace thread_setup

// run the given function on each thread (or thread_max of them) in a thread pool
// the function must take 0 arguments, or a single std::uint32_t,
// in which case the thread number [0, thread_count - 1]  will be passed to it
void do_threaded(auto func, const std::size_t thread_max = thread_count)
{
  using namespace thread_setup;

  if (thread_max > thread_count)
  {
    fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "Too many threads ({}) asked for from do_threaded (max {})\n", thread_max, thread_count);
    std::exit(EXIT_FAILURE);
  }

  for (std::uint32_t thread_index = 0; thread_index < thread_max; ++thread_index)
  {
    if constexpr (std::invocable<decltype(func)>)
      thread_functions[thread_index] = func;
    else
    {
      static_assert(std::invocable<decltype(func), std::uint32_t>, "test_repeat must be passed a function callable with 0 or 1 arguments");

      thread_functions[thread_index] = std::bind(func, thread_index);
    }

    thread_blockers[thread_index].release();
  }

  for (std::uint32_t thread_index = 0; thread_index < thread_max; ++thread_index)
    thread_completers[thread_index].acquire();
}

// call the given function on all values in the range [0, n - 1]
// uses the thread pool
// provides no guarantees on execution order or which thread functions are executed on
void loop_threaded(auto func, const std::size_t n)
{
  static_assert(std::invocable<decltype(func), std::size_t>, "test_repeat must be passed a function callable with 1 arguments");

  std::atomic<std::size_t> global_index{0};

  auto do_next_index = [&]()
  {
    while (true)
    {
      const auto index = global_index.fetch_add(1, std::memory_order_relaxed);

      if (index >= n)
        break;

      func(index);
    }
  };

  if (n < thread_count)
    do_threaded(do_next_index, n);
  else
    do_threaded(do_next_index);
}


namespace thread_setup
{
// this exists to register the atexit behaviour at startup
// thus this variable is meaningless and should never be used
bool register_terminate_threads_at_startup = []
{
  if (std::atexit([]
    {
      terminate_threads = true;

      do_threaded([](auto){ });
    })
      )
    throw "registration of thread terminating function failed";

  return true;
}();
} // namespace thread_setup

