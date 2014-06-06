function [U]=hfmm3dpart(iprec,zk,nsource,source,ifcharge,charge,ifdipole,dipstr,dipvec,ifpot,iffld,ntarget,target,ifpottarg,iffldtarg)
%HFMM3DPART Helmholtz particle target FMM in R^3.
%
% Helmholtz FMM in R^3: evaluate all pairwise particle
% interactions (ignoring self-interactions) and interactions with targets.
%
% [U]=HFMM3DPART(IPREC,ZK,NSOURCE,SOURCE,...
%         IFCHARGE,CHARGE,IFDIPOLE,DIPSTR,DIPVEC);
%
% [U]=HFMM3DPART(IPREC,ZK,NSOURCE,SOURCE,...
%         IFCHARGE,CHARGE,IFDIPOLE,DIPSTR,DIPVEC,IFPOT,IFFLD);
%
% [U]=HFMM3DPART(IPREC,ZK,NSOURCE,SOURCE,...
%         IFCHARGE,CHARGE,IFDIPOLE,DIPSTR,DIPVEC,IFPOT,IFFLD,...
%         NTARGET,TARGET);
%
% [U]=HFMM3DPART(IPREC,ZK,NSOURCE,SOURCE,...
%         IFCHARGE,CHARGE,IFDIPOLE,DIPSTR,DIPVEC,IFPOT,IFFLD,...
%         NTARGET,TARGET,IFPOTTARG,IFFLDTARG);
%
%
% This subroutine evaluates the Helmholtz potential and field due
% to a collection of charges and dipoles. We use (exp(ikr)/r) for the 
% Green's function, without the (1/4 pi) scaling. 
% Self-interactions are not-included.
%
% Input parameters:
% 
% iprec - FMM precision flag
%
%             -2 => tolerance =.5d0   =>  
%             -1 => tolerance =.5d-1  =>  1 digit 
%              0 => tolerance =.5d-2  =>  2 digits
%              1 => tolerance =.5d-3  =>  3 digits
%              2 => tolerance =.5d-6  =>  6 digits
%              3 => tolerance =.5d-9  =>  9 digits
%              4 => tolerance =.5d-12 => 12 digits
%              5 => tolerance =.5d-15 => 15 digits
%
% zk - complex Helmholtz parameter
% nsource - number of sources
% source - real (3,nsource): source locations
% ifcharge - charge computation flag
%
%         0 => do not compute
%         1 => include charge contribution
% 
% charge - complex (nsource): charge strengths 
% ifdipole - dipole computation flag
%
%         0 => do not compute
%         1 => include dipole contributions
% 
% dipole - complex (nsource): dipole strengths
% dipvec - real (3,source): dipole orientation vectors
%
% ifpot - potential computation flag, 1 => compute the potential, otherwise no
% iffld - field computation flag, 1 => compute the field, otherwise no
%
% ntarget - number of targets
% target - real (3,ntarget): target locations
%
% ifpottarg - target potential computation flag, 
%      1 => compute the target potential, otherwise no
% iffldtarg - target field computation flag, 
%      1 => compute the target field, otherwise no
%
% Output parameters: 
%
% U.pot - complex (nsource) - potential at source locations
% U.fld - complex (3,nsource) - field (i.e. -gradient) at source locations
% U.pottarg - complex (ntarget) - potential at target locations
% U.fldtarg - complex (3,ntarget) - field (i.e. -gradient) at target locations
%
% U.ier - error return code
%
%             ier=0     =>  normal execution
%             ier=4     =>  cannot allocate tree workspace
%             ier=8     =>  cannot allocate bulk FMM  workspace
%             ier=16    =>  cannot allocate mpole expansion workspace in FMM
%

if( nargin == 9 ) 
  ifpot = 1;
  iffld = 1;
  ntarget = 0;
  target = zeros(3,1);
  ifpottarg = 0;
  iffldtarg = 0;
end

if( nargin == 11 ) 
  ntarget = 0;
  target = zeros(3,1);
  ifpottarg = 0;
  iffldtarg = 0;
end

if( nargin == 13 ) 
  ifpottarg = 1;
  iffldtarg = 1;
end

ifcharge = double(ifcharge); ifdipole = double(ifdipole);
ifpot = double(ifpot); iffld = double(iffld);
ifpottarg = double(ifpottarg); iffldtarg = double(iffldtarg);

pot=0;
fld=zeros(3,1);
pottarg=0;
fldtarg=zeros(3,1);

if( ifpot == 1 ), pot=zeros(1,nsource)+1i*zeros(1,nsource); end;
if( iffld == 1 ), fld=zeros(3,nsource)+1i*zeros(3,nsource); end;
if( ifpottarg == 1 ), pottarg=zeros(1,ntarget)+1i*zeros(1,ntarget); end;
if( iffldtarg == 1 ), fldtarg=zeros(3,ntarget)+1i*zeros(3,ntarget); end;

ier=0;

if( ntarget == 0 ) 
mex_id_ = 'hfmm3dpartself(io int[x], i int[x], i dcomplex[x], i int[x], i double[xx], i int[x], i dcomplex[], i int[x], i dcomplex[], i double[xx], i int[x], io dcomplex[], i int[x], io dcomplex[])';
[ier, pot, fld] = fmm3d_r2012a(mex_id_, ier, iprec, zk, nsource, source, ifcharge, charge, ifdipole, dipstr, dipvec, ifpot, pot, iffld, fld, 1, 1, 1, 1, 3, nsource, 1, 1, 3, nsource, 1, 1);
else
mex_id_ = 'hfmm3dparttarg(io int[x], i int[x], i dcomplex[x], i int[x], i double[xx], i int[x], i dcomplex[], i int[x], i dcomplex[], i double[xx], i int[x], io dcomplex[], i int[x], io dcomplex[], i int[x], i double[], i int[x], io dcomplex[], i int[x], io dcomplex[])';
[ier, pot, fld, pottarg, fldtarg] = fmm3d_r2012a(mex_id_, ier, iprec, zk, nsource, source, ifcharge, charge, ifdipole, dipstr, dipvec, ifpot, pot, iffld, fld, ntarget, target, ifpottarg, pottarg, iffldtarg, fldtarg, 1, 1, 1, 1, 3, nsource, 1, 1, 3, nsource, 1, 1, 1, 1, 1);
end

if( ifpot == 1 ), U.pot=pot; end
if( iffld == 1 ), U.fld=fld; end
if( ifpottarg == 1 ), U.pottarg=pottarg; end
if( iffldtarg == 1 ), U.fldtarg=fldtarg; end
U.ier=ier;


