module mod_usr
    use mod_hd
    implicit none

contains

    subroutine usr_init()
        usr_init_one_grid => initonegrid_usr

        call set_coordinate_system("spherical")
        call hd_activate()

    end subroutine usr_init

    subroutine initonegrid_usr(ixI^L,ixO^L,w,x)
        use mod_physics
        integer, intent(in) :: ixI^L, ixO^L
        double precision, intent(in) :: x(ixI^S,1:ndim)
        double precision, intent(inout) :: w(ixI^S,1:nw)

        double precision :: rbs,xc^D,xcc^D
        double precision :: xcart(ixI^S,1:ndim)
        logical, save    :: first = .true.

        if (first) then
            if (mype==0) then
                print *,'Sedov-Taylor'
            end if
            first=.false.
        end if

        rbs=0.05d0
        w(ixO^S,rho_)=1.d0
        w(ixO^S,p_)=0.00001d0
        where(x(ixO^S,1)<rbs)
            w(ixO^S,p_)=1000.d0
        endwhere
        w(ixO^S,mom(:))=0.d0

        call phys_to_conserved(ixI^L,ixO^L,w,x)

    end subroutine initonegrid_usr
end module mod_usr
