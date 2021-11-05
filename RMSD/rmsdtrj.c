#include <float.h>
#include "rmsdf.h"
#include "pdb.h"

float rmsdcalc(float xyz1[][3],float xyz2[][3], int n){	
	float mov_com[3];
	float mov_to_ref[3];
	float U[3][3];
	float rmsd;
	//calculate_rotation_rmsd(xyz1,xyz2,n,mov_com,mov_to_ref,U,&rmsd);
	fast_rmsd(xyz1,xyz2,n,&rmsd);	
	return rmsd;
}

void usage(char* program){
	printf("Usage:\n");
	printf("\t%s traj1.pdb traj2.pdb\n",program);
}

int main(int argc, char **argv)
{
	char *trj1,*trj2;
	FILE *fp1,*fp2;
	int n1,n2,m1,m2;
	float rmsd;
	if (argc!=3){
		usage(argv[0]);
                exit(EXIT_FAILURE);	
	}
	trj1=argv[1];
	trj2=argv[2];
	n1=CountAtoms(trj1);
	n2=CountAtoms(trj2);
	//printf("n1:%d n2:%d\n",n1,n2);
	
	if (n1!=n2){
		fprintf(stderr,"Error, Number of Atoms is different in %s: %d, and %s: %d",trj1,n1,trj2,n2);
	}
	float xyz1[n1][3];
	float xyz2[n1][3];

	if ((fp1=fopen(trj1, "r")) == NULL){
		fprintf(stderr,"FILE NOT FOUND: %s",argv[1]);
                exit(EXIT_FAILURE);
        }
	if ((fp2=fopen(trj2, "r")) == NULL){
                fprintf(stderr,"FILE NOT FOUND: %s",argv[1]);
                exit(EXIT_FAILURE);
        }
	int i=0;
	for(;;){
		m1=set_coord(fp1,xyz1,n1);
			int j=0;
			for(;;){
				m2=set_coord(fp2,xyz2,n1);
		//printf("m1:%d m2:%d\n",m1,m2);
				if (m1==n1 && m2==n1){
					rmsd=rmsdcalc(xyz1,xyz2,n1);
					//if (rmsd <=6){
					printf("%d\t%d\t%f\n", i,j,rmsd);
					//}
				}else{
					break;
				}
			j++;
			}
		printf("\n");
	//	fseek(fp2, 0, SEEK_SET);
	//	i++;
		if (m1 != 0){
			fseek(fp2, 0, SEEK_SET);
			i++;
		//	continue;
			}
		else
			break;
		}
	fclose(fp1);
	fclose(fp2);
	exit(EXIT_SUCCESS);
}
