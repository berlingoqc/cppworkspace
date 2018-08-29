#include "../../../include/capture.hpp"


int main() {

    String f = "lol.jpg";

    ImageWrapper img(f);
    if(!img.Open()) {
        std::cerr << "failed" << std::endl;
    }

    img.ShowAndWait();

    img.SharpenImageFilter2D();

    img.ShowAndWait();


    return 0;
}