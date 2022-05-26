subroutine interpolate_2d(xp, x, y, y_out, fill_value, m, n_in, n_out)
    implicit none
    real(8), intent(in) :: xp(m, n_out), x(m, n_in), y(m, n_in)
    integer, intent(in) :: m, n_in, n_out
    real(8), intent(in) :: fill_value
    real(8), intent(out) :: y_out(m, n_out)
    ! locals
    integer :: i, j, k
    real(8) :: weight

    y_out = fill_value

    do i=1,m
        do j=1,n_out
            ! search for lower and upper indices
            do k=1,n_in - 1
                if  ((x(i, k) <= xp(i, j)) .and. (xp(i, j) < x(i, k+1))) then
                    weight = (xp(i, j) - x(i, k))/(x(i, k + 1) - x(i, k))
                    y_out(i, j) = y(i, k) * (1-weight) + y(i, k + 1) * weight
                else if (x(i, k) == xp(i, j)) then
                    y_out(i, j) = y(i, k)
                else if (x(i, k + 1) == xp(i, j)) then
                    y_out(i, j) = y(i, k + 1)
                end if
            end do
        end do
    end do
end subroutine