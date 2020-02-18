program runDL_vs_Exposure
	use params
	use util
	use like
	implicit none

	integer :: i,ns,nm,n_ex,cnt,verbose
	double precision :: m_min,m_max,sigma_min,sigma_max,ex_min,ex_max
	double precision,dimension(:),allocatable :: m_vals,ex_vals,DL_ex
	double precision,dimension(:),allocatable :: DL,sigma_p_vals
	character(len=10) :: junk1,junk2,junk3
	character(len=100) :: filename1,inp
  CHARACTER(len=1) :: arg


	cnt = COMMAND_ARGUMENT_COUNT()
	if(cnt.eq.0) then
  	write(*,*)'ERROR, AT LEAST ONE COMMAND-LINE ARGUMENTS REQUIRED, STOPPING'
  	stop
	end if

  CALL get_command_argument(1, inp)
	if (cnt.eq.2) then
		CALL get_command_argument(2, arg)
		read(arg,*) verbose
	else
		verbose = 0
	end if

	if ((verbose.ne.0).and.(verbose.ne.1)) then
		write(*,*)'ERROR, VERBOSE MUST BE 0 OR 1, STOPPING'
		stop
	end if

	! Seed rand and random
	call itime(mytime)
	call cpu_time(clock_start)


	! Resolution of floor
	sigma_min = 1.0d-52
	sigma_max = 1.0d-42
	ns = 500

	! Read in signal distribution
	open(unit=111,file=trim(homedir)//'/data/recoils/RD_sig_'//trim(inp)//'.txt')
	read(111,*) junk1,n_ex,nTot_bins
	allocate(RD_sig(n_ex,nTot_bins))
	allocate(RD_wimp(nTot_bins))
	allocate(ex_vals(n_ex))
	do i = 1,n_ex
		read(111,*) ex_vals(i),RD_sig(i,:)
	end do
	close(111)

	! Read in background distribution
	open(unit=222,file=trim(homedir)//'/data/recoils/RD_bg_'//trim(inp)//'.txt')
	read(222,*) junk1,n_bg,nTot_bins
	allocate(RD_bg(nTot_bins,n_bg))
	allocate(R_bg(n_bg))
	allocate(R_bg_err(n_bg))
	do i = 1,n_bg
		read(222,*) R_bg(i),R_bg_err(i),RD_bg(:,i)
	end do
	close(222)



	! write neutrino floor
	filename1 = trim(mylimitsdir)//'/DL1D_'//trim(inp)//'.txt'
 	open(unit=1000,file=trim(filename1))
	nm = 1
	allocate(DL(nm))
	write(*,*) '====NuFloor===='
	write(*,*) 'Reading '//trim(inp)
	write(*,*) 'nTot_bins =',nTot_bins
	write(*,*) 'n_nu =',n_bg
	write(*,*) 'n_ex =',n_ex
	write(*,*) 'R_bg_err = ',R_bg_err
	write(*,*) 'R_bg = ',R_bg
	do i = 1,n_ex
		Exposure = ex_vals(i)
		RD_sig = RD_sig*Exposure
		RD_bg = RD_bg*Exposure
		write(*,*) i,'of',n_ex,'Exposure = ',Exposure,'ton-year'
		call DiscoveryLimit((/6.0d0/),1,sigma_min,sigma_max,ns,verbose,DL)
		write(1000,*) DL
		RD_sig = RD_sig/Exposure
		RD_bg = RD_bg/Exposure
	end do
	close(1000)

	call cpu_time(clock_stop); write(*,*) 'Time elapsed = ',clock_stop-clock_start
	write(*,*) '=====DONE====='
	stop
end program runDL_vs_Exposure
