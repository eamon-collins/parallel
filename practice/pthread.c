#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define num_threads 16

void *avg (void *d);

typedef struct arg {
	int tnum;
} t_arg;

float *arr[2];
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t end_barrier;
pthread_barrier_t iter_barrier;

int main(int argc, char *argv[]){
	time_t start_time = time(NULL);
	pthread_barrier_init(&end_barrier, NULL, num_threads+1);
	pthread_barrier_init(&iter_barrier, NULL, num_threads);
	
	arr[0] = (float *) malloc(num_threads * sizeof(float));
	arr[1] = (float *) malloc(num_threads * sizeof(float));
	
	int i;
	for(i = 0; i < num_threads; i++){
		arr[0][i] = (float)i;
	}
	pthread_t tids[num_threads];
	t_arg args[num_threads];
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	for(i=0; i < num_threads; i++){
		args[i].tnum = i;
		pthread_create(&tids[i], &attr, avg, &args[i]);
	}
	pthread_barrier_wait(&end_barrier);
	for(i = 0; i < num_threads; i++){
		pthread_join(tids[i], NULL);
	}
	printf("end fo real\n");
	for(i = 0; i < num_threads; i++){
		printf("%f ", arr[0][i]);
	}

	time_t end_time = time(NULL);
	printf("\nexec time: %f", difftime(end_time, start_time));
	pthread_exit(NULL);
}

void *avg(void *d){
	int curr_cells_index = 0, next_cells_index = 1;
	t_arg *arg = (t_arg*) d;
	int iters = 10000;
	int i;
	for(i=0; i < iters; i++){
		pthread_mutex_lock(&lock);
		if (arg->tnum == 0){
			arr[next_cells_index][arg->tnum] = (arr[curr_cells_index][arg->tnum] + arr[curr_cells_index][arg->tnum +1]) / 2;
		}else if (arg->tnum == num_threads -1){
			arr[next_cells_index][arg->tnum] = (arr[curr_cells_index][arg->tnum -1] + arr[curr_cells_index][arg->tnum]) / 2;
		}else{
			arr[next_cells_index][arg->tnum] = (arr[curr_cells_index][arg->tnum -1] + arr[curr_cells_index][arg->tnum] + arr[curr_cells_index][arg->tnum +1]) / 3;
		}
		pthread_mutex_unlock(&lock);
		curr_cells_index = next_cells_index;
		next_cells_index = !curr_cells_index;
		pthread_barrier_wait(&iter_barrier);
	}
	pthread_barrier_wait(&end_barrier);
}

