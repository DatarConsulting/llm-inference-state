// Single-threaded in-memory encode benchmark. Store every byte in a preallocated
// buffer; checksum it after timing. No file I/O, decoder or inference in the timer.
#include "row-format.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using namespace gdn_local;
int main(int argc,char** argv) {
    if(argc!=2) return 1;
    std::ifstream f(argv[1],std::ios::binary|std::ios::ate);
    const auto bytes=f.tellg();
    if(bytes<=0 || bytes%65536!=0) return 2;
    f.seekg(0);std::vector<float> data(bytes/4);
    if(!f.read(reinterpret_cast<char*>(data.data()),bytes)) return 3;
    const size_t num=data.size()/elements;
    std::vector<uint8_t> out(num*(4+max_encoded));
    std::array<Policy,4> methods={Policy::midpoint,Policy::median,Policy::optimal,Policy::screened};
    const char* names[]={"mid","median","opt","screened"};
    std::cout<<"repeat,method,seconds,raw_bytes,encoded_bytes,checksum\n";
    for(int trial=-1;trial<8;++trial) for(int m=0;m<4;++m) {
        int k=(m+trial+4)%4;size_t end=0;
        auto start=std::chrono::steady_clock::now();
        for(size_t h=0;h<num;++h) {
            uint8_t* frame=out.data()+end;size_t pos=5;frame[4]=head_tag;
            bool nonzero=false;
            for(int r=0;r<128;++r) {
                size_t n=encode_row(data.data()+h*elements+r*128,frame+pos,methods[k]);
                nonzero=nonzero || ((frame[pos]&7)!=4);
                pos+=n;
            }
            if(!nonzero) pos=4;
            put32(frame,uint32_t(pos-4));end+=pos;
        }
        std::chrono::duration<double> dt=std::chrono::steady_clock::now()-start;
        uint64_t sum=0;for(size_t j=0;j<end;++j) sum=sum*31+out[j];
        if(trial>=0) std::cout<<trial<<','<<names[k]<<','<<dt.count()<<','<<bytes<<','<<end<<','<<sum<<'\n';
    }
}
