#pragma once
#define ARMA_MAX_ELEM 0x200000000ULL
#define ARMA_64BIT_WORD
#include <armadillo>
#include "xatu/SystemConfiguration.hpp"

namespace xatu {

class HDF5Configuration : public SystemConfiguration {

    private:
        std::string filename;
    
    public:
        HDF5Configuration(std::string);
        ~HDF5Configuration(){};

    private:
        void parseContent();
};

}