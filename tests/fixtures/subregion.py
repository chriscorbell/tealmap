# Writes a region file containing only the chunks in a local box, for test fixtures.
import struct,sys
src,dst,x0,x1,z0,z1=sys.argv[1],sys.argv[2],*map(int,sys.argv[3:7])
data=open(src,'rb').read()
loc=bytearray(4096); ts=bytearray(4096); body=bytearray()
sector=2
for lz in range(z0,z1+1):
    for lx in range(x0,x1+1):
        i=lx+lz*32
        off=struct.unpack('>I',b'\0'+data[i*4:i*4+3])[0]
        if off==0: continue
        start=off*4096
        n=struct.unpack('>I',data[start:start+4])[0]
        chunk=data[start:start+4+n]
        pad=(-len(chunk))%4096
        cnt=(len(chunk)+pad)//4096
        struct.pack_into('>I',loc,i*4,(sector<<8)|cnt)
        ts[i*4:i*4+4]=data[4096+i*4:4096+i*4+4]
        body+=chunk+b'\0'*pad; sector+=cnt
open(dst,'wb').write(bytes(loc)+bytes(ts)+bytes(body))
print(dst, sector-2, 'sectors')
