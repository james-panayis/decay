#include <fmt/format.h>
#include <fmt/compile.h>
#include <fmt/color.h>

#include <sys/random.h>

#include <random>
#include <array>


template<class T>
concept arithmetic = std::floating_point<T> || std::integral<T>;



// Create a seed sequence with enough seeds to fully initialize a std::mt19937
[[nodiscard]] std::seed_seq generate_seeds() noexcept
{
  std::array<std::mt19937::result_type, std::mt19937::state_size> seeds;

  auto it = seeds.begin();

  auto result = getrandom(&(*it), seeds.end() - it, 0);

  if (result == -1)
    fmt::print("getrandom failed, returned {}\n", result);

  return std::seed_seq(seeds.begin(), seeds.end());
}


thread_local std::mt19937 prng_ = []{ auto seeds = generate_seeds(); return std::mt19937{seeds}; }();


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

  for (int i = 0; i < int{N}; ++i)
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

  double average_d_p_mag_K {0};
  double average_d_pt_K    {0};
  double average_d_p_mag_pi{0};
  double average_d_pt_pi   {0};
  //std::array<double, 3> average_p_pi{0, 0, 0};

  constexpr int repeats = 10'000'000;

  int back_count = 0;

  for (int i = 0; i < repeats; ++i)
  {
    // angle of kaon motion in B meson's reference frame
    const double phi   =           std::uniform_real_distribution<double>{0.0 , 2.0 * std::numbers::pi}(prng_);
    const double theta = std::acos(std::uniform_real_distribution<double>{-1.0, 1.0                   }(prng_));

    // unscaled velocity of kaon motion in B meson's reference frame
    const std::array<double, 3> v_temp{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};

    // velocities of decay products in B meson's reference frame in d
    const std::array<double, 3> v_K { v_mag_K  * v_temp};
    const std::array<double, 3> v_pi{-v_mag_pi * v_temp};

    //fmt::print("zero?: {}\n\n", (std::sqrt(pow<2>(p_mag*c) + pow<2>(m_pi)*pow<4>(c)) + std::sqrt(pow<2>(p_mag*c) + pow<2>(m_K)*pow<4>(c))) / pow<2>(c));

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

    const std::array<double, 3> d_p_K  = d_g_K  * m_K  * d_v_K;
    const std::array<double, 3> d_p_pi = d_g_pi * m_pi * d_v_pi;

    average_d_p_mag_K  += mag(d_p_K);
    average_d_pt_K     += std::sqrt(pow<2>(d_p_K [0]) + pow<2>(d_p_K [1]));
    average_d_p_mag_pi += mag(d_p_pi);
    average_d_pt_pi    += std::sqrt(pow<2>(d_p_pi[0]) + pow<2>(d_p_pi[1]));


    if (i < 100)//(d_v_pi[2] < 0.0)
    {
      fmt::print("gamma: {}\n", d_g_K);
      fmt::print("d_v_K:  {},   mag: {}\n", d_v_K , mag(d_v_K ));
      fmt::print("d_p_pi: {},   mag: {}\n", d_p_pi, mag(d_p_pi));
    }
    if (d_v_pi[2] < 0.0)
      ++back_count;
  }

  fmt::print("\n");

  average_d_p_mag_K  *= 1.0 / static_cast<double>(repeats);
  average_d_pt_K     *= 1.0 / static_cast<double>(repeats);
  average_d_p_mag_pi *= 1.0 / static_cast<double>(repeats);
  average_d_pt_pi    *= 1.0 / static_cast<double>(repeats);

  fmt::print("average_d_pt_pi   : {}\n", average_d_pt_pi/pow<6>(10)/c);
  fmt::print("average_d_pt_K    : {}\n", average_d_pt_K /pow<6>(10)/c);
  fmt::print("average_d_p_mag_pi: {}\n", average_d_p_mag_pi/pow<6>(10)/c);
  fmt::print("average_d_p_mag_K : {}\n", average_d_p_mag_K /pow<6>(10)/c);

  fmt::print("fraction going backwards: {}\n", static_cast<double>(back_count)/static_cast<double>(repeats));

  return EXIT_SUCCESS;
}
