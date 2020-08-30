#include <bits/stdc++.h>
using namespace std;


extern "C" void axpb_cpp(double a[], double b[], double c[], int n){
  int i;

  for(i=0; i<n; i++)
    {
      c[i] += a[i] + b[i];
    }
}
