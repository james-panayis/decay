#include <fmt/format.h>
#include <fmt/compile.h>
#include <fmt/color.h>

#include <thread>
#include <functional>
#include <semaphore>


// Provides general threading infrastructure and helper functions


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


std::vector<std::function<void()>> thread_functions(thread_count);

std::array thread_blockers = 
  ( []<std::size_t... I>(std::index_sequence<I...>)
    { return std::array<std::binary_semaphore, sizeof...(I)>{{ ((void) I, std::binary_semaphore(0))... }}; }
  )(std::make_index_sequence<100>{});

std::array thread_completers = 
  ( []<std::size_t... I>(std::index_sequence<I...>)
    { return std::array<std::binary_semaphore, sizeof...(I)>{{ ((void) I, std::binary_semaphore(0))... }}; }
  )(std::make_index_sequence<100>{});

void thread_loop(const std::uint32_t thread_index)
{
  while (true)
  {
    thread_blockers[thread_index].acquire();

    thread_functions[thread_index]();

    thread_completers[thread_index].release();
  }
}

std::vector<std::thread> threads = []
{
  std::vector<std::thread> out;

  out.reserve(thread_count);

  for (std::size_t thread_index = 0; thread_index < thread_count; ++thread_index)
    out.emplace_back(thread_loop, thread_index);

  return out;
}();

// run the given function on each thread in a thread pool
// the function must take 0 arguments, or a single std::uint32_t,
// in which case the thread number [0, thread_count - 1]  will be passed
void do_threaded(auto func, const std::size_t thread_max = thread_count)
{
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

