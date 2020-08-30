/* #include <conio.h> */

int double_input(int input) {
  return input * 2;
}


void axpb_c(double a[], double b[], double c[], int n)
{
  int i;

  for(i=0; i<n; i++)
    {
      c[i] += a[i] + b[i];
    }
}
