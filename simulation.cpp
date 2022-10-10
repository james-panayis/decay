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
[[nodiscard]] consteval auto pow(auto x) -> decltype(x)
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

  // magnitude of momentum and speed of B in B reference frame
  //constexpr double p_B_mag = 50'000'000'000.0 * c;
  //constexpr double v_mag_B {std::sqrt(pow<2>(p_B_mag) / (pow<2>(m_B ) + pow<2>(p_B_mag/c)))};

  fmt::print("momentum of decay products: {}  speed of K: {}  speed of pi: {}\n\n", p_mag, v_mag_K, v_mag_pi);

  std::array<double, 3> average_v_K {0, 0, 0};
  std::array<double, 3> average_v_pi{0, 0, 0};

  constexpr int repeats = 1'000'000;

  for (int i = 0; i < repeats; ++i)
  {
    // angle of kaon motion in B meson's reference frame
    const double phi   =           std::uniform_real_distribution<double>{0.0 , 2.0 * std::numbers::pi}(prng_);
    const double theta = std::acos(std::uniform_real_distribution<double>{-1.0, 1.0                   }(prng_));

    // unscaled velocity of kaon motion in B meson's reference frame
    const std::array<double, 3> v{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};

    // velocities of decay products in B meson's reference frame in d
    const std::array<double, 3> v_K { v_mag_K  * v};
    const std::array<double, 3> v_pi{-v_mag_pi * v};

    average_v_K  += v_K;
    average_v_pi += v_pi;

    //fmt::print("zero?: {}\n\n", (std::sqrt(pow<2>(p_mag*c) + pow<2>(m_pi)*pow<4>(c)) + std::sqrt(pow<2>(p_mag*c) + pow<2>(m_K)*pow<4>(c))) / pow<2>(c));
  }

  average_v_K  *= 1.0 / static_cast<double>(repeats);
  average_v_pi *= 1.0 / static_cast<double>(repeats);

  fmt::print("average_v_K : {} {} {}\n"  , average_v_K [0], average_v_K [1], average_v_K [2]);
  fmt::print("average_v_pi: {} {} {}\n\n", average_v_pi[0], average_v_pi[1], average_v_pi[2]);

  return EXIT_SUCCESS;
}
