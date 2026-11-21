// Row layout adapted from the archived local GDN research codec (November 2026).
#pragma once
#include <algorithm>
#include "reference.hpp"
#include <cstdint>
#include <cstring>
#include <stdexcept>

// Linux little-endian research format. All transformations are on words.
namespace gdn_local {
constexpr size_t elements = 16384;
constexpr size_t max_encoded = 1 + 128*513;
constexpr uint8_t head_tag = 3;
inline unsigned width(uint32_t x) { return x ? 32u-unsigned(__builtin_clz(x)) : 0; }
inline uint32_t word(const float * p) { uint32_t u; std::memcpy(&u,p,4); return u; }
inline void put32(uint8_t * p,uint32_t x) { std::memcpy(p,&x,4); }
inline uint32_t get32(const uint8_t * p) { uint32_t x;std::memcpy(&x,p,4);return x; }
inline uint32_t zig(int64_t d) { return uint32_t(d>=0 ? 2*d : -2*d-1); }
inline int64_t unzig(uint32_t z) { return z&1 ? -int64_t((uint64_t(z)+1)/2) : int64_t(z/2); }
inline void pack8(const uint32_t * x,unsigned b,uint8_t * out) {
    uint64_t acc=0;unsigned have=0;size_t pos=0;
    for(int i=0;i<8;++i) {
        acc|=uint64_t(x[i])<<have;have+=b;
        while(have>=8) {out[pos++]=uint8_t(acc);acc>>=8;have-=8;}
    }
}
inline void unpack8(const uint8_t * in,unsigned b,uint32_t * x) {
    uint64_t acc=0;unsigned have=0;size_t pos=0;
    const uint64_t mask=(uint64_t(1)<<b)-1;
    for(int i=0;i<8;++i) {
        while(have<b) {acc|=uint64_t(in[pos++])<<have;have+=8;}
        x[i]=uint32_t(acc&mask);acc>>=b;have-=b;
    }
}
enum class Policy { midpoint, median, optimal, screened };
struct RowStats { unsigned residual_bytes=0; unsigned alternative_bytes=0; unsigned lower_bytes=0; };
inline size_t encode_row(const float * x,uint8_t * out,Policy policy=Policy::midpoint,
                         RowStats* stats=nullptr) {
    uint32_t u[128],z[128],signs[4]={};unsigned lo[4]={255,255,255,255},hi[4]={};
    uint32_t amin=0x7fffffff,amax=0,any=0;
    for(int i=0;i<128;++i) {
        u[i]=word(x+i);any|=u[i];
        const unsigned e=(u[i]>>23)&255,t=i/32;
        lo[t]=std::min(lo[t],e);hi[t]=std::max(hi[t],e);
        amin=std::min(amin,u[i]&0x7fffffff);amax=std::max(amax,u[i]&0x7fffffff);
        signs[t]|=((u[i]^u[0])>>31)<<(i%32);
    }
    
    const unsigned row_lo=*std::min_element(lo,lo+4);
    const unsigned row_width=width(*std::max_element(hi,hi+4)-row_lo);
    const size_t row_size=386+16*row_width;
    unsigned tile_widths[4],flags=0;
    for(int t=0;t<4;++t) {
        if(signs[t]) flags|=1u<<t;
        tile_widths[t]=width(hi[t]-lo[t]);
    }
    const size_t tile_size=391+4*(tile_widths[0]+tile_widths[1]+tile_widths[2]+tile_widths[3]);
    unsigned alternative=any ? unsigned(std::min({size_t(513),row_size,tile_size})) : 1;
    unsigned header=17+4*unsigned(__builtin_popcount(flags));
    unsigned lower=header;
    if(stats || policy==Policy::screened) {
        for (int g=0;g<16;++g) {
            uint32_t low=0x7fffffff, high=0;
            for(int j=0;j<8;++j) {
                uint32_t a=u[g*8+j]&0x7fffffff;
                low=std::min(low,a); high=std::max(high,a);
            }
            lower+=width(high-low);
        }
    }
    bool sweep=policy==Policy::optimal || (policy==Policy::screened && lower<alternative);
    uint32_t magnitude=uint32_t((uint64_t(amin)+amax)/2);
    reference::Solution solution{};
    if (sweep) {
        solution=reference::solve(u);
        magnitude=solution.base;
    } else if (policy==Policy::median) {
        uint32_t sorted[128];
        for(int i=0;i<128;++i) sorted[i]=u[i]&0x7fffffff;
        std::nth_element(sorted,sorted+63,sorted+128);
        magnitude=sorted[63];
    }
    const uint32_t base=magnitude|(u[0]&0x80000000);
    unsigned widths[16];
    size_t center_size=header;
    for(int q=0;q<16;++q) {
        uint32_t both=0;
        for(int j=0;j<8;++j) {
            const int i=8*q+j;z[i]=zig(int64_t(u[i]&0x7fffffff)-int64_t(base&0x7fffffff));both|=z[i];
        }
        widths[q]=width(both);center_size+=widths[q];
    }
    if (sweep) {
        int total=0;
        for(auto b:widths) total+=static_cast<int>(b);
        if(total!=solution.width_sum) throw std::logic_error("coverage certificate mismatch");
    }
    if(stats) {
        stats->residual_bytes=static_cast<unsigned>(center_size);
        stats->lower_bytes=lower;
        stats->alternative_bytes=any ? static_cast<unsigned>(std::min({size_t(513),row_size,tile_size})) : 1;
    }
    if(!any) {out[0]=4;return 1;}
    unsigned mode=0;size_t size=513;
    if(row_size<size) {size=row_size;mode=1;}
    if(tile_size<size) {size=tile_size;mode=2;}
    if(center_size<size) {size=center_size;mode=3;}
    if(mode==0) {out[0]=0;std::memcpy(out+1,x,512);return 513;}
    size_t pos=0;
    if(mode==3) {
        out[pos++]=uint8_t(3|(flags<<3));put32(out+pos,base);pos+=4;
        for(int q=0;q<16;q+=4) {
            const uint32_t w=widths[q]|(widths[q+1]<<6)|(widths[q+2]<<12)|(widths[q+3]<<18);
            out[pos++]=uint8_t(w);out[pos++]=uint8_t(w>>8);out[pos++]=uint8_t(w>>16);
        }
        for(int t=0;t<4;++t) if(signs[t]) {put32(out+pos,signs[t]);pos+=4;}
        for(int q=0;q<16;++q) {pack8(z+8*q,widths[q],out+pos);pos+=widths[q];}
    } else {
        out[pos++]=uint8_t(mode|(mode==1?row_width<<3:0));
        if(mode==1) out[pos++]=uint8_t(row_lo);
        else {
            for(int t=0;t<4;++t) out[pos++]=uint8_t(lo[t]);
            out[pos++]=uint8_t(tile_widths[0]|(tile_widths[1]<<4));
            out[pos++]=uint8_t(tile_widths[2]|(tile_widths[3]<<4));
        }
        for(uint32_t w:u) {
            const uint32_t m=(w&0x7fffff)|((w>>8)&0x800000);
            out[pos++]=uint8_t(m);out[pos++]=uint8_t(m>>8);out[pos++]=uint8_t(m>>16);
        }
        for(int q=0;q<16;++q) {
            const unsigned t=q/4,b=mode==1?row_width:tile_widths[t],l=mode==1?row_lo:lo[t];
            for(int j=0;j<8;++j) z[j]=((u[8*q+j]>>23)&255)-l;
            pack8(z,b,out+pos);pos+=b;
        }
    }
    if(pos!=size) throw std::logic_error("local row size mismatch");
    return pos;
}
inline size_t decode_row(const uint8_t * in,size_t n,float * x) {
    if(!n || in[0]&128) throw std::runtime_error("invalid local row");
    const unsigned tag=in[0],mode=tag&7;size_t pos=1;
    auto need=[&](size_t k) {if(k>n-pos) throw std::runtime_error("truncated local row");};
    if(mode==0) {if(tag)throw std::runtime_error("raw row flags");need(512);std::memcpy(x,in+pos,512);return 513;}
    if(mode==4) {if(tag!=4)throw std::runtime_error("zero row flags");std::memset(x,0,512);return 1;}
    uint32_t u[128],z[8];
    if(mode==3) {
        need(16);const uint32_t base=get32(in+pos);pos+=4;unsigned widths[16],flags=tag>>3;uint32_t signs[4]={};
        for(int q=0;q<16;q+=4) {
            uint32_t w=uint32_t(in[pos])|(uint32_t(in[pos+1])<<8)|(uint32_t(in[pos+2])<<16);pos+=3;
            for(int j=0;j<4;++j) {widths[q+j]=(w>>(6*j))&63;if(widths[q+j]>32)throw std::runtime_error("invalid local width");}
        }
        for(int t=0;t<4;++t) if(flags&(1u<<t)) {need(4);signs[t]=get32(in+pos);pos+=4;if(!signs[t])throw std::runtime_error("empty sign mask");}
        for(int q=0;q<16;++q) {
            need(widths[q]);unpack8(in+pos,widths[q],z);pos+=widths[q];
            for(int j=0;j<8;++j) {
                const int i=q*8+j;const int64_t a=int64_t(base&0x7fffffff)+unzig(z[j]);
                if(a<0 || a>0x7fffffff)throw std::runtime_error("invalid local magnitude");
                u[i]=uint32_t(a)|((base&0x80000000)^(((signs[i/32]>>(i%32))&1)<<31));
            }
        }
    } else if(mode==1 || mode==2) {
        unsigned lo[4],b[4];
        if(mode==1) {need(1);for(int t=0;t<4;++t){lo[t]=in[pos];b[t]=tag>>3;}++pos;}
        else {
            if(tag!=2)throw std::runtime_error("tile row flags");need(6);
            for(int t=0;t<4;++t)lo[t]=in[pos++];
            for(int t=0;t<4;t+=2) {b[t]=in[pos]&15;b[t+1]=in[pos++]>>4;}
        }
        for(unsigned w:b)if(w>8)throw std::runtime_error("invalid exponent width");
        need(384);const uint8_t * mant=in+pos;pos+=384;
        for(int q=0;q<16;++q) {
            const unsigned t=q/4;need(b[t]);unpack8(in+pos,b[t],z);pos+=b[t];
            for(int j=0;j<8;++j) {
                int i=q*8+j;const unsigned e=lo[t]+z[j];if(e>255)throw std::runtime_error("invalid exponent");
                const uint32_t m=uint32_t(mant[3*i])|(uint32_t(mant[3*i+1])<<8)|(uint32_t(mant[3*i+2])<<16);
                u[i]=(m&0x7fffff)|((m&0x800000)<<8)|(e<<23);
            }
        }
    } else throw std::runtime_error("unknown local row mode");
    std::memcpy(x,u,512);return pos;
}
inline size_t encode(const float * x,uint8_t * out) {
    out[0]=head_tag;size_t pos=1;uint32_t any=0;
    for(int r=0;r<128;++r) {pos+=encode_row(x+128*r,out+pos);for(int i=0;i<128;++i)any|=word(x+128*r+i);}
    return any?pos:0;
}
inline void decode(const uint8_t * in,size_t n,float * x) {
    if(!n){std::memset(x,0,elements*4);return;}
    if(n>max_encoded || in[0]!=head_tag)throw std::runtime_error("invalid local head");
    size_t pos=1;
    for(int r=0;r<128;++r)pos+=decode_row(in+pos,n-pos,x+128*r);
    if(pos!=n)throw std::runtime_error("trailing local head bytes");
}
}
