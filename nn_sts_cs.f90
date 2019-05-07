module evrdata
implicit none
real*8, allocatable, dimension (:,:) :: rrovibener, provibener, dummyarray1, &
rqmax, rqmin, rtperiod, rbheight, dummyarray2, dummyarray3, dummyarray4, dummyarray5, &
pqmax, pqmin, ptperiod, pbheight
integer :: rmaxvib, rmaxrot, pmaxvib, pmaxrot
real*8 :: rde, pde

contains

subroutine readener
implicit none
integer :: nv, nj, np
integer :: iv, ij, ii
real*8 :: wt 

nv=100
nj=400
open(unit=101,file="N2_rovib.dat",status="old") !product
open(unit=102,file="NO_rovib.dat",status="old") !reactant
 
allocate (dummyarray1(0:nv,0:nj), dummyarray2(0:nv,0:nj),dummyarray3(0:nv,0:nj),dummyarray4(0:nv,0:nj), dummyarray5(0:nv,0:nj))
dummyarray1=-10.0d0
dummyarray2=-10.0d0
dummyarray3=-10.0d0
dummyarray4=-10.0d0

!no of states and dissociation energy of the product diatom
read(101,*)np,pde
pmaxvib=0
pmaxrot=0
do ii = 1, np
! v', j', weight of the state (dummy), E(v',j'), q+, q-, centrifugal barrier height, vibrational time period
  read(101,*)iv,ij,wt,dummyarray1(iv,ij),dummyarray2(iv,ij),dummyarray3(iv,ij),dummyarray4(iv,ij),dummyarray5(iv,ij)
  if (iv>pmaxvib)pmaxvib=iv
  if (ij>pmaxrot)pmaxrot=ij
end do

allocate (provibener(0:pmaxvib,0:pmaxrot),pqmax(0:pmaxvib,0:pmaxrot),&
pqmin(0:pmaxvib,0:pmaxrot),ptperiod(0:pmaxvib,0:pmaxrot),pbheight(0:pmaxvib,0:pmaxrot))

provibener(0:pmaxvib,0:pmaxrot)=dummyarray1(0:pmaxvib,0:pmaxrot)
pqmax(0:pmaxvib,0:pmaxrot)=dummyarray2(0:pmaxvib,0:pmaxrot)
pqmin(0:pmaxvib,0:pmaxrot)=dummyarray3(0:pmaxvib,0:pmaxrot)
pbheight(0:pmaxvib,0:pmaxrot)=dummyarray4(0:pmaxvib,0:pmaxrot)
ptperiod(0:pmaxvib,0:pmaxrot)=dummyarray5(0:pmaxvib,0:pmaxrot)

deallocate (dummyarray1,dummyarray2,dummyarray3,dummyarray4,dummyarray5)

allocate (dummyarray1(0:nv,0:nj), dummyarray2(0:nv,0:nj),dummyarray3(0:nv,0:nj),dummyarray4(0:nv,0:nj), dummyarray5(0:nv,0:nj))
dummyarray1=-10.0d0
dummyarray2=-10.0d0
dummyarray3=-10.0d0
dummyarray4=-10.0d0

!no of states and dissociation energy of the reactant diatom
read(102,*)np,rde
rmaxvib=0
rmaxrot=0
do ii = 1, np
! v, j, weight of the state (dummy), E(v,j), q+, q-, centrifugal barrier height, vibrational time period
  read(102,*)iv,ij,wt,dummyarray1(iv,ij),dummyarray2(iv,ij),dummyarray3(iv,ij),dummyarray4(iv,ij),dummyarray5(iv,ij)
  if (iv>rmaxvib)rmaxvib=iv
  if (ij>rmaxrot)rmaxrot=ij
end do

allocate (rrovibener(0:rmaxvib,0:rmaxrot),rqmax(0:rmaxvib,0:rmaxrot),&
rqmin(0:rmaxvib,0:rmaxrot),rtperiod(0:rmaxvib,0:rmaxrot),rbheight(0:rmaxvib,0:rmaxrot))

rrovibener(0:rmaxvib,0:rmaxrot)=dummyarray1(0:rmaxvib,0:rmaxrot)
rqmax(0:rmaxvib,0:rmaxrot)=dummyarray2(0:rmaxvib,0:rmaxrot)
rqmin(0:rmaxvib,0:rmaxrot)=dummyarray3(0:rmaxvib,0:rmaxrot)
rbheight(0:rmaxvib,0:rmaxrot)=dummyarray4(0:rmaxvib,0:rmaxrot)
rtperiod(0:rmaxvib,0:rmaxrot)=dummyarray5(0:rmaxvib,0:rmaxrot)

deallocate (dummyarray1,dummyarray2,dummyarray3,dummyarray4,dummyarray5)

end subroutine
end module

program nn_sts
use evrdata
implicit none
!for N+NO system
real*8, parameter :: pi = acos(-1.0d0), amu_to_au = 1822.88862d0
real*8,parameter :: ma=14.007d0*amu_to_au,mb=ma,mc=15.9994d0*amu_to_au
real*8 :: epcol, ecol, vr, vp, mur, mup, dele, weight, energy, inirovibener, inivener,&
inirotener, vel, qp, qm, bh, tp, totcs, ics, decol, pecol
integer, parameter :: nh0n=24, nh1n=24, nh2n=24, nh3n=24, nh4n=24, nh5n=24, nh6n=24, nh7n=24, &
ninp=24, nh8n=24, noutp=1, bwvib=2, bwrot=3
real*8, dimension (:,:) :: w0(ninp,nh0n), w1(nh0n,nh1n), w2(nh1n,nh2n),  w3(nh2n,nh3n), w4(nh3n,nh4n), &
w5(nh4n,nh5n), w6(nh5n,nh6n),  w7(nh6n,nh7n), w8(nh7n,nh8n), wo(nh8n,noutp)
real*8, dimension (:) :: b0(nh0n), b1(nh1n), b2(nh2n), b3(nh3n), b4(nh4n), b5(nh5n), b6(nh6n), b7(nh7n), b8(nh8n),&
bo(noutp), ainp(ninp), aout(noutp), scaleainp(ninp)
real*8, dimension (:) :: h0out(nh0n), h1out(nh1n),  h2out(nh2n), h3out(nh3n), h4out(nh4n), &
h5out(nh5n), h6out(nh6n),  h7out(nh7n), h8out(nh8n), mval(ninp), stdv(ninp)
real*8 :: cssum, totwt, fy, ewt, state_prime
real*8, allocatable, dimension(:) :: wnorm
integer, allocatable, dimension(:) :: vibar, rotar
integer :: ii, jj, inivib, inirot, indvib, indrot, kk, tnsts, ll, mm
character (len=24) :: outfile

mur= ma*(mb+mc)/(ma+mb+mc)
mup= mc*(ma+mb)/(ma+mb+mc)

call readener

dele=pde-rde

open(unit=10,file="Coeff_h0W.dat")
open(unit=11,file="Coeff_h1W.dat")
open(unit=12,file="Coeff_h2W.dat")
open(unit=13,file="Coeff_h3W.dat")
open(unit=14,file="Coeff_h4W.dat")
open(unit=15,file="Coeff_h5W.dat")
open(unit=16,file="Coeff_h6W.dat")
open(unit=17,file="Coeff_h7W.dat")

open(unit=20,file="Coeff_h0b.dat")
open(unit=21,file="Coeff_h1b.dat")
open(unit=22,file="Coeff_h2b.dat")
open(unit=23,file="Coeff_h3b.dat")
open(unit=24,file="Coeff_h4b.dat")
open(unit=25,file="Coeff_h5b.dat")
open(unit=26,file="Coeff_h6b.dat")
open(unit=27,file="Coeff_h7b.dat")

open(unit=31,file="Coeff_outW.dat")
open(unit=32,file="Coeff_outb.dat")

open(unit=41,file="Coeff_mval.txt")
open(unit=42,file="Coeff_stdv.txt")

do ii =1,ninp
  read(10,*)(w0(ii,jj),jj=1,nh0n)
end do

do ii =1,nh0n
  read(20,*)b0(ii)
end do

do ii =1,nh0n
  read(11,*)(w1(ii,jj),jj=1,nh1n)
end do

do ii =1,nh1n
  read(21,*)b1(ii)
end do

do ii =1,nh1n
  read(12,*)(w2(ii,jj),jj=1,nh2n)
end do

do ii =1,nh2n
  read(22,*)b2(ii)
end do

do ii =1,nh2n
  read(13,*)(w3(ii,jj),jj=1,nh3n)
end do

do ii =1,nh3n
  read(23,*)b3(ii)
end do

do ii =1,nh3n
  read(14,*)(w4(ii,jj),jj=1,nh4n)
end do

do ii =1,nh4n
  read(24,*)b4(ii)
end do

do ii =1,nh4n
  read(15,*)(w5(ii,jj),jj=1,nh5n)
end do

do ii =1,nh5n
  read(25,*)b5(ii)
end do

do ii =1,nh5n
  read(16,*)(w6(ii,jj),jj=1,nh6n)
end do

do ii =1,nh6n
  read(26,*)b6(ii)
end do

do ii =1,nh6n
  read(17,*)(w7(ii,jj),jj=1,nh7n)
end do

do ii =1,nh7n
  read(27,*)b7(ii)
end do

do ii =1,nh8n
  read(31,*)(wo(ii,jj),jj=1,noutp)
end do

do ii =1,noutp
  read(32,*)bo(ii)
end do

do ii =1,ninp
  read(41,*)mval(ii)
end do

do ii =1,ninp
  read(42,*)stdv(ii)
end do

print*,"v, j, Etra, v', j'"
read*,inivib, inirot, Ecol, indvib, indrot

  ainp(1) = rrovibener(inivib,inirot)
  ainp(2) = rrovibener(inivib,0)
  ainp(3) = dfloat(inivib)
  ainp(4) = rrovibener(inivib,inirot) - rrovibener(inivib,0)
  ainp(5) = dfloat(inirot)
  ainp(6) = sqrt(dfloat(inirot*(inirot+1)))
  ainp(7) = ecol
  ainp(8) = sqrt(2.0d0*ecol/27.2114d0/mur)
  ainp(9) = rqmax(inivib,inirot)
  ainp(10) = rqmin(inivib,inirot)
  ainp(11) = rbheight(inivib,inirot)*27.2114d0-rde
  ainp(12) = rtperiod(inivib,inirot)

  ainp(13) = provibener(indvib,indrot)
  ainp(14) = provibener(indvib,0)
  ainp(15) = dfloat(indvib)
  ainp(16) = provibener(indvib,indrot) - provibener(indvib,0)
  ainp(17) = dfloat(indrot)
  ainp(18) = sqrt(dfloat(indrot*(indrot+1)))
  ainp(19) = dele+ainp(7)+ainp(1)-ainp(13)
  ainp(20) = sqrt(2.0d0*ainp(19)/27.2114d0/mup)
  ainp(21) = pqmax(indvib,indrot)
  ainp(22) = pqmin(indvib,indrot)
  ainp(23) = pbheight(indvib,indrot)*27.2114d0-pde
  ainp(24) = ptperiod(indvib,indrot)

  if (ainp(1) < -1.0d0 .or. ainp(13) <-1.0d0 .or. ainp(19) <0.0d0) then
    aout=0.0d0
  else

  scaleainp = ainp
  do ii = 1, ninp
    if (stdv(ii) >0.0d0 ) then
      scaleainp(ii) = (scaleainp(ii) - mval(ii))/stdv(ii)
    else
      scaleainp(ii) = scaleainp(ii) - mval(ii)
    end if
  end do

   h0out = asinh(matmul(scaleainp,w0)+b0)*1.256734802399369d0
   h1out = max(0.0d0,matmul(h0out,w1)+b1)+scaleainp
   h2out = asinh(matmul(h1out,w2)+b2)*1.256734802399369d0
   h3out = max(0.0d0,matmul(h2out,w3)+b3)+h1out
   h4out = asinh(matmul(h3out,w4)+b4)*1.256734802399369d0
   h5out = max(0.0d0,matmul(h4out,w5)+b5)+h3out
   h6out = asinh(matmul(h5out,w6)+b6)*1.256734802399369d0
   h7out = max(0.0d0,matmul(h6out,w7)+b7)+h5out
   aout = 0.4d0/(1.0d0+exp(-1.0d0*(matmul(h7out,wo)+bo)))
  end if
  if (aout(1)<1d-5) aout(1)=0.0d0
  write(*,*)aout(1)
end program
