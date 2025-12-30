#include <complex>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

/****************************************************************
 * Utility template that helps us find the offset in a 1d vector
 * when we are using a (x,y) coordinate.
 */
template <const uint32_t channel_count>
auto offset_in_interleaved_1d_vec(const uint32_t width,
                                  const uint32_t x,
                                  const uint32_t y,
                                  const uint32_t channel) -> size_t
{
    const auto offset = (y * width + x) * channel_count + channel;
    return offset;
}

/*****************************************************************/
auto get_number_of_iterations(const std::complex<double>& z0, const double size, const int max)
    -> int
{
    auto z = z0;
    for (auto i = 0; i < max; i++)
    {
        if (std::abs(z) > size)
            return i;

        z = (z * z) + z0;
    }

    // default to max
    return max;
}

/****************************************************************
 * From Wikipedia ( https://en.wikipedia.org/wiki/Mandelbrot_set#Formal_definition ):
 * `The Mandelbrot set is the set of values of c in the complex plane for which the orbit of the
 * critical point z = 0 under iteration of the quadratic map remains bounded.` z{n+1} = z^2{n} + c
 */
auto create_grayscale_mandelbrot_image(const double center_x,
                                       const double center_y,
                                       const double size,
                                       const int    max_iterations,
                                       const int    pixels_wide) -> std::vector<std::uint8_t>
{
    // A vector to hold the grayscale image data, pixels_wide^2 in size, and initialized with zero
    auto image = std::vector<std::uint8_t>(pixels_wide * pixels_wide, 0);

    // Convenience lambda
    auto get_scaled_coordinate = [&](const double center, const double xy) {
        return (center - (size / 2) + ((size * xy) / pixels_wide));
    };

    for (auto y = 0; y < pixels_wide; y++)
    {
        for (auto x = 0; x < pixels_wide; x++)
        {
            // Scale the x/y coordinate to be within the size x size box
            auto x0 = get_scaled_coordinate(center_x, x);
            auto y0 = get_scaled_coordinate(center_y, y);

            // Find out how many iterations (of the function) we can go through before the complex
            // number becomes unstable
            auto z    = std::complex<double>(x0, y0);
            auto gray = max_iterations - get_number_of_iterations(z, size, max_iterations);

            // Get the offset, using the x,y coordinate, to our memory position in the 1D vector
            auto offset = offset_in_interleaved_1d_vec<1>(pixels_wide, x, y, 0);

            // Now save the grayscale value
            image[offset] = gray;
        }
    }

    return image;
}

/*****************************************************************
 * Convenience function to create the full path to our output image
 */
auto get_output_file_path() -> std::string
{
    const auto current_path = std::filesystem::current_path();
    const auto path         = current_path / std::filesystem::path("mandelbrot.jpg");
    const auto full_path    = std::filesystem::weakly_canonical(path);
    return full_path.string();
}

/*****************************************************************/
int main(int, char**)
{
    // The center x,y for the Mandelbrot box
    const auto center_x = -0.5; // center x
    const auto center_y = 0.0;  // center y

    // The size of the Mandelbrot box, which in this case, is a 2x2 box
    const auto size = 2.0;

    // The number of iterations we will allow for, before aborting
    const auto max_iterations = 255;

    const auto image_pixels_wide = 512 * 2;

    // Create the image data
    auto grayscale_image = create_grayscale_mandelbrot_image(
        center_x, center_y, size, max_iterations, image_pixels_wide);

    // Convert the image data to an OpenCV Mat
    auto output_gray =
        cv::Mat(image_pixels_wide, image_pixels_wide, CV_8UC1, grayscale_image.data());

    // Write out the data to disk; first up, let's get a path to write to
    const auto full_path  = get_output_file_path();
    const auto successful = cv::imwrite(full_path.c_str(), output_gray);
    if (!successful)
        std::cout << "Failed to write out file\n";
    else
        std::cout << "Success!\n";
}