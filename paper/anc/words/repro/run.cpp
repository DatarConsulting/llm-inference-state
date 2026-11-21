// Replay archived binary32 words. This executable performs no floating-point
// arithmetic on them. A head frame consists of a length and the row records.
#include "row-format.hpp"
#include <array>
#include <fstream>
#include <iostream>
#include <string>

using namespace gdn_local;
using Head = std::array<float,elements>;
using Bytes = std::vector<uint8_t>;
struct RowResult { size_t n; int tag; RowStats stats; };

static Bytes encode_and_check(const Head& head, Policy policy,
                              std::array<RowResult,128>& rows) {
    Bytes frame(4 + max_encoded);
    size_t pos=5;
    frame[4]=head_tag;
    bool nonzero=false;
    for(int r=0;r<128;++r) {
        auto& result=rows[r];
        size_t n=encode_row(head.data()+128*r,frame.data()+pos,policy,&result.stats);
        result.tag=frame[pos]&7;
        result.n=n;
        nonzero=nonzero || result.tag!=4;
        pos+=n;
    }
    if(!nonzero) pos=4;
    frame.resize(pos);
    put32(frame.data(),static_cast<uint32_t>(pos-4));
    Head decoded{};
    decode(frame.data()+4,get32(frame.data()),decoded.data());
    if(std::memcmp(head.data(),decoded.data(),65536)) throw std::logic_error("round trip");
    return frame;
}

int main(int argc,char** argv) {
    if(argc!=4) {std::cerr<<"Usage: replay heads.u32 rows.csv heads.csv\n";return 2;}
    uint32_t one=1;
    if(*reinterpret_cast<uint8_t*>(&one)!=1 || sizeof(float)!=4) return 3;
    std::ifstream in(argv[1],std::ios::binary);
    std::ofstream rows(argv[2]), heads(argv[3]);
    if(!in || !rows || !heads) return 4;
    rows<<"id,row,mid,median,opt,mid_res,median_res,opt_res,alt,lower,mid_mode,median_mode,opt_mode\n";
    heads<<"id,mid,median,opt\n";
    Head data{};
    int id=0;
    while(in.read(reinterpret_cast<char*>(data.data()),65536)) {
        std::array<std::array<RowResult,128>,3> results{};
        std::array<size_t,3> sizes{};
        int k=0;
        for(Policy policy:{Policy::midpoint,Policy::median,Policy::optimal}) {
            sizes[k]=encode_and_check(data,policy,results[k]).size();
            ++k;
        }
        std::array<RowResult,128> guarded{};
        auto guard_frame=encode_and_check(data,Policy::screened,guarded);
        std::array<RowResult,128> direct{};
        if (guard_frame!=encode_and_check(data,Policy::optimal,direct))
            throw std::logic_error("screened frame disagrees");
        heads<<id<<','<<sizes[0]<<','<<sizes[1]<<','<<sizes[2]<<'\n';
        for(int r=0;r<128;++r) {
            auto a=results[0][r], b=results[1][r], c=results[2][r];
            if(c.n>std::min(a.n,b.n) || c.stats.residual_bytes>std::min(a.stats.residual_bytes,b.stats.residual_bytes))
                throw std::logic_error("suboptimal result");
            rows<<id<<','<<r<<','<<a.n<<','<<b.n<<','<<c.n<<','
                <<a.stats.residual_bytes<<','<<b.stats.residual_bytes<<','<<c.stats.residual_bytes
                <<','<<a.stats.alternative_bytes<<','<<a.stats.lower_bytes<<','<<a.tag<<','<<b.tag<<','<<c.tag<<'\n';
        }
        ++id;
    }
    if(!in.eof() || in.gcount()!=0 || !rows || !heads || !id) return 5;
    std::cout<<id<<" heads verified with midpoint, median, optimal and screened encoders\n";
}
