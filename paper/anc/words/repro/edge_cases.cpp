// An independent finite-domain oracle for the production single-anchor solver.
#include "row-format.hpp"
#include <array>
#include <iostream>
#include <random>

int main() {
    for (unsigned rec=0;rec<4096;++rec) {
        std::array<uint32_t,128> w{};
        for (int j=0;j<128;++j)
            w[j]=(rec>>(3*((j/8)%2*2+j%2)))&7;
        int best=512;
        for(int64_t p=0;p<8;++p) {
            int sum=0;
            for(int g=0;g<16;++g) {
                uint32_t top=0;
                for(int j=0;j<8;++j) {
                    int64_t d=int64_t(w[8*g+j])-p;
                    top=std::max(top,uint32_t(d<0?-2*d-1:2*d));
                }
                while(top) {++sum; top>>=1;}
            }
            best=std::min(best,sum);
        }
        if(reference::solve(w.data()).width_sum!=best) return 1;
    }
    std::mt19937 rng(20261121);
    uint32_t edge[]={0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fa00001,0xffc12345,0x007fffff,0x00800000,0x7f7fffff};
    bool tags[5]={};
    for(int r=0;r<128;++r) {
        std::array<uint32_t,128> u{};
        for(int j=0;j<128;++j) {
            switch(r%8) {
            case 0: break;
            case 1: u[j]=edge[j%11];break;
            case 2: u[j]=rng();break;
            case 3: u[j]=0x80000000;break;
            case 4: u[j]=0x3f800000+(rng()%16);break;
            case 5: u[j]=((100+(rng()%4))<<23)|(rng()&0x807fffff);break;
            case 6: u[j]=((10+50*(j/32)+(rng()%4))<<23)|(rng()&0x807fffff);break;
            default:u[j]=edge[r%11];break;
            }
        }
        float src[128],dst[128];std::memcpy(src,u.data(),512);
        for(auto p:{gdn_local::Policy::midpoint,gdn_local::Policy::median,gdn_local::Policy::optimal,gdn_local::Policy::screened}) {
            uint8_t bytes[513];size_t n=gdn_local::encode_row(src,bytes,p);
            tags[bytes[0]&7]=true;
            if(gdn_local::decode_row(bytes,n,dst)!=n || std::memcmp(src,dst,512)) return 2;
        }
    }
    for(bool active:tags) if(!active) return 3;
    std::cout<<"4096 compiled-solver oracles; 512 row round trips; all five row modes exercised\n";
}
