#include "timeblit/random.hpp"

#include <fmt/format.h>
#include <fmt/compile.h>
#include <fmt/color.h>

#include <random>
#include <array>
#include <thread>
#include <atomic>
#include <iostream>
#include <fstream>

using namespace movency;


// x to the power of non-negative integer P
template<std::size_t P>
[[nodiscard]] constexpr auto pow(auto x) -> decltype(x)
{
  if constexpr (P == 0)
    return 1;
  else if constexpr (P % 2 == 0)
    return pow<P/2>(x) * pow<P/2>(x);
  else
    return x * pow<P/2>(x) * pow<P/2>(x);
}


// multiply array by value
template<arithmetic T, std::size_t N>
[[nodiscard]] constexpr auto operator* (const T mult, const std::array<T, N> arr)
{
  std::array<T, N> out;

  for (std::uint32_t i = 0; i < N; ++i)
  {
    out[i] = mult * arr[i];
  }

  return out;
}

template<arithmetic T, std::size_t N>
[[nodiscard]] constexpr auto operator* (const std::array<T, N> arr, const T mult)
{
  return mult * arr;
}

template<arithmetic T, std::size_t N>
constexpr auto operator*= (std::array<T, N>& arr, const T mult)
{
  arr = mult * arr;

  return &arr;
}


// add to array
template<arithmetic T, std::size_t N>
constexpr auto operator+= (std::array<T, N>& lhs, const std::array<T, N> rhs)
{
  for (int i = 0; i < int{N}; ++i)
  {
    lhs[i] += rhs[i];
  }

  return &lhs;
}


// calculate the length of a vector
template<arithmetic T, std::size_t N>
constexpr T mag(const std::array<T, N>& arr) noexcept
{
  T out = 0;

  for (auto val : arr)
    out += pow<2>(val);

  return std::sqrt(out);
}


// allow printing of vectors
template<>
struct fmt::formatter<std::array<double, 3>>
{
  constexpr auto parse(fmt::format_parse_context& ctx)
  {
    return ctx.end();
  }

  template<class FormatContext>
  auto format(const std::array<double, 3> arr, FormatContext& ctx)
  {
    return fmt::format_to(ctx.out(), "{} {} {}", arr[0], arr[1], arr[2]);
  }
};


int main()
{
  fmt::print("starting\n\n");

  // speed of light in m/s
  constexpr double c{299'792'458};

  // masses of particles in eV/c^2
  constexpr double m_B {5'279'660'000};
  constexpr double m_K {  493'667'000};
  constexpr double m_pi{  139'570'390};

  // magnitude of momentum that each decay product has in the B meson's reference frame in eV/c
  constexpr double p_mag = c * std::sqrt(pow<4>(m_B) + pow<4>(m_K) + pow<4>(m_pi) - 2*pow<2>(m_B*m_K) - 2*pow<2>(m_B*m_pi) - 2*pow<2>(m_K*m_pi)) / (2 * m_B);

  // speeds of decay products in B meson's reference frame in c
  constexpr double v_mag_K {std::sqrt(pow<2>(p_mag) / (pow<2>(m_K ) + pow<2>(p_mag/c)))};
  constexpr double v_mag_pi{std::sqrt(pow<2>(p_mag) / (pow<2>(m_pi) + pow<2>(p_mag/c)))};

  // magnitude of momentum and speed of B meson in detector's reference frame
  constexpr double d_p_B_mag {50'000'000'000.0 * c};
  constexpr double d_v_mag_B {std::sqrt(pow<2>(d_p_B_mag) / (pow<2>(m_B) + pow<2>(d_p_B_mag/c)))};
  [[maybe_unused]] constexpr std::array<double, 3> d_v_B{0.0, 0.0, d_v_mag_B};

  fmt::print("momentum of decay products in B frame: {}  speed of K: {}  speed of pi: {}\n", p_mag, v_mag_K, v_mag_pi);
  fmt::print("speed of B0 in reference frame of detector: {}\n", d_v_mag_B);

  // gamma of B in detector's frame
  constexpr double d_g_B = 1.0 / std::sqrt(1.0 - pow<2>(d_v_mag_B/c));

  // lifetime of B meson in each reference frame
  constexpr double B_lifetime{0.000000000001519};
  constexpr double d_B_lifetime{B_lifetime * d_g_B};

  // distance travelled by B meson before decaying in detector's frame
  constexpr double d_d_B{d_B_lifetime * d_v_mag_B};

  fmt::print("average distance travelled by B meson in detector's frame: {}\n\n", d_d_B);

  int back_count = 0;

  constexpr std::uint32_t bucket_count = 4000;

  constexpr double bs_hist_d_p_mag_K {60000.0 / double{bucket_count}};
  constexpr double bs_hist_d_p_mag_pi{60000.0 / double{bucket_count}};
  constexpr double bs_hist_d_pt_K {3000.0 / double{bucket_count}};
  constexpr double bs_hist_d_pt_pi{3000.0 / double{bucket_count}};
  std::array<std::atomic<std::uint64_t>, bucket_count> hist_d_p_mag_K{};
  std::array<std::atomic<std::uint64_t>, bucket_count> hist_d_p_mag_pi{};
  std::array<std::atomic<std::uint64_t>, bucket_count> hist_d_pt_K{};
  std::array<std::atomic<std::uint64_t>, bucket_count> hist_d_pt_pi{};
  std::atomic<double> average_d_p_mag_K {0};
  std::atomic<double> average_d_p_mag_pi{0};
  std::atomic<double> average_d_pt_K    {0};
  std::atomic<double> average_d_pt_pi   {0};

  constexpr double bs_hist_impact_parameter_K {0.005 / double{bucket_count}};
  constexpr double bs_hist_impact_parameter_pi{0.005 / double{bucket_count}};
  std::array<std::atomic<std::uint64_t>, bucket_count> hist_impact_parameter_K{};
  std::array<std::atomic<std::uint64_t>, bucket_count> hist_impact_parameter_pi{};
  std::atomic<double> average_impact_parameter_K {0};
  std::atomic<double> average_impact_parameter_pi{0};

  std::atomic<std::int64_t> repeats = 0;

  auto loop = [&](std::stop_token stoken)
  {
    std::array<std::uint64_t, bucket_count> hist_d_p_mag_K_temp{};
    std::array<std::uint64_t, bucket_count> hist_d_p_mag_pi_temp{};
    std::array<std::uint64_t, bucket_count> hist_d_pt_K_temp{};
    std::array<std::uint64_t, bucket_count> hist_d_pt_pi_temp{};
    std::array<std::uint64_t, bucket_count> hist_impact_parameter_K_temp{};
    std::array<std::uint64_t, bucket_count> hist_impact_parameter_pi_temp{};
    double average_d_p_mag_K_temp {0};
    double average_d_p_mag_pi_temp{0};
    double average_d_pt_K_temp    {0};
    double average_d_pt_pi_temp   {0};
    double average_impact_parameter_K_temp {0};
    double average_impact_parameter_pi_temp{0};

    int back_count_temp{0};

    std::int64_t repeats_temp{0};

    while (!stoken.stop_requested())
    {
      for (int i = 0; i < 1000; ++i)
      {
        ++repeats_temp;

        // angle of kaon motion in B meson's reference frame
        const double phi   =           random::fast(random::uniform_distribution{ 0.0, 2.0 * std::numbers::pi});
        const double theta = std::acos(random::fast(random::uniform_distribution{-1.0, 1.0                   }));

        // unscaled velocity of kaon motion in B meson's reference frame
        const std::array<double, 3> v_temp{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};

        // velocities of decay products in B meson's reference frame in d
        const std::array<double, 3> v_K { v_mag_K  * v_temp};
        const std::array<double, 3> v_pi{-v_mag_pi * v_temp};

        const double v = d_v_mag_B;

        // velocities of decay products in detector's reference frame in c
        const std::array<double, 3> d_v_K {std::sqrt(1 - pow<2>(v/c)) * v_K [0] / (1 + v/pow<2>(c) * v_K [2]),
                                           std::sqrt(1 - pow<2>(v/c)) * v_K [1] / (1 + v/pow<2>(c) * v_K [2]),
                                           (v_K[2] + v)                         / (1 + v/pow<2>(c) * v_K [2]) };

        const std::array<double, 3> d_v_pi{std::sqrt(1 - pow<2>(v/c)) * v_pi[0] / (1 + v/pow<2>(c) * v_pi[2]),
                                           std::sqrt(1 - pow<2>(v/c)) * v_pi[1] / (1 + v/pow<2>(c) * v_pi[2]),
                                           (v_pi[2] + v)                        / (1 + v/pow<2>(c) * v_pi[2]) };

        const double d_g_K  = 1.0 / std::sqrt(1.0 - pow<2>(mag(d_v_K) /c));
        const double d_g_pi = 1.0 / std::sqrt(1.0 - pow<2>(mag(d_v_pi)/c));

        const std::array<double, 3> d_p_K  = (d_g_K  * m_K ) * d_v_K;
        const std::array<double, 3> d_p_pi = (d_g_pi * m_pi) * d_v_pi;

        const double d_p_mag_K  = mag(d_p_K);
        const double d_p_mag_pi = mag(d_p_pi);
        const double d_pt_K     = std::sqrt(pow<2>(d_p_K [0]) + pow<2>(d_p_K [1]));
        const double d_pt_pi    = std::sqrt(pow<2>(d_p_pi[0]) + pow<2>(d_p_pi[1]));

        average_d_p_mag_K_temp  += d_p_mag_K;
        average_d_p_mag_pi_temp += d_p_mag_pi;
        average_d_pt_K_temp     += d_pt_K;
        average_d_pt_pi_temp    += d_pt_pi;

        ++hist_d_p_mag_K_temp [static_cast<std::size_t>(d_p_mag_K  / (bs_hist_d_p_mag_K * c * pow<6>(10)))];
        ++hist_d_p_mag_pi_temp[static_cast<std::size_t>(d_p_mag_pi / (bs_hist_d_p_mag_pi * c * pow<6>(10)))];
        ++hist_d_pt_K_temp    [static_cast<std::size_t>(d_pt_K     / (bs_hist_d_pt_K * c * pow<6>(10)))];
        ++hist_d_pt_pi_temp   [static_cast<std::size_t>(d_pt_pi    / (bs_hist_d_pt_pi * c * pow<6>(10)))];

        const double impact_parameter_K  = d_d_B * std::sqrt(pow<2>(d_v_K [0]) + pow<2>(d_v_K [1])) / mag(d_v_K);
        const double impact_parameter_pi = d_d_B * std::sqrt(pow<2>(d_v_pi[0]) + pow<2>(d_v_pi[1])) / mag(d_v_pi);

        average_impact_parameter_K_temp  += impact_parameter_K;
        average_impact_parameter_pi_temp += impact_parameter_pi;

        ++hist_impact_parameter_K_temp [static_cast<std::size_t>(impact_parameter_K  / bs_hist_impact_parameter_K)];
        ++hist_impact_parameter_pi_temp[static_cast<std::size_t>(impact_parameter_pi / bs_hist_impact_parameter_pi)];

        if (d_v_pi[2] < 0.0)
          ++back_count_temp;
      }

      for (std::uint32_t i = 0; i < bucket_count; ++i)
      {
        hist_d_p_mag_K[i]  += hist_d_p_mag_K_temp[i];
        hist_d_p_mag_pi[i] += hist_d_p_mag_pi_temp[i];
        hist_d_pt_K[i]     += hist_d_pt_K_temp[i];
        hist_d_pt_pi[i]    += hist_d_pt_pi_temp[i];
        hist_impact_parameter_K[i]  += hist_impact_parameter_K_temp[i];
        hist_impact_parameter_pi[i] += hist_impact_parameter_pi_temp[i];
      }

      average_d_p_mag_K  += average_d_p_mag_K_temp;
      average_d_p_mag_pi += average_d_p_mag_pi_temp;
      average_d_pt_K     += average_d_pt_K_temp;
      average_d_pt_pi    += average_d_pt_pi_temp;
      average_impact_parameter_K  += average_impact_parameter_K_temp;
      average_impact_parameter_pi += average_impact_parameter_pi_temp;

      back_count += back_count_temp;
      repeats    += repeats_temp;
    }

    for (std::uint32_t i = 0; i < bucket_count; ++i)
    {
      hist_d_p_mag_K[i]  += hist_d_p_mag_K_temp[i];
      hist_d_p_mag_pi[i] += hist_d_p_mag_pi_temp[i];
      hist_d_pt_K[i]     += hist_d_pt_K_temp[i];
      hist_d_pt_pi[i]    += hist_d_pt_pi_temp[i];
      hist_impact_parameter_K[i]  += hist_impact_parameter_K_temp[i];
      hist_impact_parameter_pi[i] += hist_impact_parameter_pi_temp[i];
    }

    average_d_p_mag_K  += average_d_p_mag_K_temp;
    average_d_p_mag_pi += average_d_p_mag_pi_temp;
    average_d_pt_K     += average_d_pt_K_temp;
    average_d_pt_pi    += average_d_pt_pi_temp;
    average_impact_parameter_K  += average_impact_parameter_K_temp;
    average_impact_parameter_pi += average_impact_parameter_pi_temp;

    back_count += back_count_temp;
    repeats    += repeats_temp;
  };

  {
    std::vector<std::jthread> loop_threads{};

    for (std::uint32_t i = 1; i < std::thread::hardware_concurrency(); ++i)
      loop_threads.emplace_back(loop);

    // wait for input
    fmt::print("Looping on {} threads. Type then press enter to stop.", loop_threads.size());

    int temp; std::cin >> temp;
  }

  fmt::print("\n");

  auto normalize = [&repeats] (auto& val)
  {
    val.store(val.load() / static_cast<double>(repeats.load()));
  };

  normalize(average_d_p_mag_K);
  normalize(average_d_pt_K);
  normalize(average_d_p_mag_pi);
  normalize(average_d_pt_pi);
  normalize(average_impact_parameter_K);
  normalize(average_impact_parameter_pi);

  fmt::print("average_d_pt_pi   : {}\n", average_d_pt_pi/pow<6>(10)/c);
  fmt::print("average_d_pt_K    : {}\n", average_d_pt_K /pow<6>(10)/c);
  fmt::print("average_d_p_mag_pi: {}\n", average_d_p_mag_pi/pow<6>(10)/c);
  fmt::print("average_d_p_mag_K : {}\n", average_d_p_mag_K /pow<6>(10)/c);

  fmt::print("\n");

  fmt::print("average_impact_parameter_K : {}\n", average_impact_parameter_K);
  fmt::print("average_impact_parameter_pi: {}\n", average_impact_parameter_pi);

  fmt::print("\nfraction going backwards: {}\n", static_cast<double>(back_count)/static_cast<double>(repeats));

  fmt::print("\nrepeats: {}\n", repeats);

  std::ofstream out;

  out.open("cache/out.csv");

  for (std::uint32_t i = 0; i < bucket_count; ++i)
  {
    std::array bucket_sizes = {&bs_hist_d_p_mag_K, &bs_hist_d_p_mag_pi, &bs_hist_d_pt_K, &bs_hist_d_pt_pi, &bs_hist_impact_parameter_K, &bs_hist_impact_parameter_pi};

    std::array histograms = {&hist_d_p_mag_K, &hist_d_p_mag_pi, &hist_d_pt_K, &hist_d_pt_pi, &hist_impact_parameter_K, &hist_impact_parameter_pi};

    for (std::uint32_t j = 0; j < histograms.size(); ++j)
    {
      out << fmt::format("{},", *bucket_sizes[j] * (static_cast<double>(i) + 0.5));
      out << fmt::format("{},", (*histograms[j])[i]);
    }

    out << "\n";
  }
  
  return EXIT_SUCCESS;
}
