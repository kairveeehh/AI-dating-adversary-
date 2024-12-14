#include <bits/stdc++.h>
using namespace std ;

int main(){
    printf("HELLO PPL!!!\n") ;
    printf("WELCOME TO KAIRVEE'S MULTITHREADED SERVER!!!\n") ;
    return 0 ;
}







#include <bits/stdc++.h>
#include <sys/socket.h>
#include <limits.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
// #include "myQueue.h"


#define SERVERPORT 8989
#define BUFSIZE 4096
#define SOCKETERROR (-1)
#define SERVER_BACKLOG 100
#define THREAD_POOL_SIZE 20

pthread_t thread_pool[THREAD_POOL_SIZE] ;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER ;
pthread_cond_t cond_var = PTHREAD_COND_INITIALIZER ;

typedef struct sockaddr_in SA_IN ;
typedef struct sockaddr SA ;

void * handle_connection(void * p_clinet_soccket) ;
int check(int exp , const char *msg) ;
void * thread_function(void * args) ;

int main(int argc , char **argv){
    int server_socket , client_socket, addr_size ;
    SA_IN server_addr , client_addr ;

    // for(int i =0 ; i < THREAD_POOL_SIZE ; i ++){
    //     pthread_create(&thread_pool[i] , NULL , thread_function , NULL) ;
    // }

    check((server_socket = socket(AF_INET , SOCK_STREAM , 0)),
            "Failed to create socket") ;
    
    server_addr.sin_family = AF_INET ;
    server_addr.sin_addr.s_addr = INADDR_ANY ;
    server_addr.sin_port = htons(SERVERPORT) ;

    check((bind(server_socket, (SA*)&server_addr , sizeof(server_addr))),
            "Bind Failed!") ;
    check(listen(server_socket, SERVER_BACKLOG),
            "Listen Failed!") ;
    
    while(true){
        printf("Waiting for connection...\n") ;
        addr_size = sizeof(SA_IN) ;
        check(client_socket = accept(server_socket , (SA*)&client_addr, (socklen_t*)&addr_size),
                "accept failed!!") ;
        printf("Connected\n") ;
    
        // pthread_t t ;
        int *pclient = (int *)malloc(sizeof(int)) ;
        *pclient = client_socket ;
        handle_connection(pclient) ;
        // this enqueue is causing the race condition.
        // pthread_mutex_lock(&lock) ;
        // enqueue(pclient) ;
        // pthread_mutex_unlock(&lock) ;
        // pthread_create(&t , NULL , handle_connection , (void *) pclient) ;
        

    }

    return 0 ;

}

int check(int exp, const char *msg) {
    if(exp == SOCKETERROR){
        perror(msg) ;
        exit(1) ;
    }
    return exp ;
}




void * handle_connection(void * p_clinet_soccket) {
    int clinet_soccket = *((int *) p_clinet_soccket) ;
    free(p_clinet_soccket) ;
    char buffer[BUFSIZE] ;
    size_t bytes_read ;
    int msgsize = 0 ;
    char actualpath[PATH_MAX + 1] ;

    while((bytes_read = read(clinet_soccket , buffer + msgsize , sizeof(buffer) - msgsize - 1)) > 0){
        msgsize += bytes_read ;
        if(msgsize > BUFSIZE - 1 || buffer[msgsize-1] == '\n') break ;
    }
    check(bytes_read , "recv error") ;
    buffer[msgsize - 1] = 0 ;

    printf("REQUEST: %s\n" , buffer) ;
    fflush(stdout) ;

    if(realpath(buffer , actualpath) == NULL){
        printf("ERROR(bad path): %s\n" , buffer) ;
        close(clinet_soccket) ;
        return NULL;
    }
    FILE *fp = fopen(actualpath , "r") ;

    if(fp == NULL){
        printf("ERROR(open): %s\n" , buffer) ;
        close(clinet_soccket) ;
        return NULL;
    }

    while((bytes_read = fread(buffer , 1 , BUFSIZE , fp)) > 0){
        printf("sending %zu bytes\n" , bytes_read) ;
        write(clinet_soccket , buffer , bytes_read) ;
    }
    close(clinet_soccket) ;
    fclose(fp) ;
    printf("CLosing connection\n") ;
    return NULL ;
}
#include <bits/stdc++.h>
#include <sys/socket.h>
#include <limits.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
// #include "myQueue.h"


#define SERVERPORT 8989
#define BUFSIZE 4096
#define SOCKETERROR (-1)
#define SERVER_BACKLOG 100
#define THREAD_POOL_SIZE 20

pthread_t thread_pool[THREAD_POOL_SIZE] ;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER ;
pthread_cond_t cond_var = PTHREAD_COND_INITIALIZER ;

typedef struct sockaddr_in SA_IN ;
typedef struct sockaddr SA ;

void * handle_connection(void * p_clinet_soccket) ;
int check(int exp , const char *msg) ;
void * thread_function(void * args) ;

int main(int argc , char **argv){
    int server_socket , client_socket, addr_size ;
    SA_IN server_addr , client_addr ;

    // for(int i =0 ; i < THREAD_POOL_SIZE ; i ++){
    //     pthread_create(&thread_pool[i] , NULL , thread_function , NULL) ;
    // }

    check((server_socket = socket(AF_INET , SOCK_STREAM , 0)),
            "Failed to create socket") ;
    
    server_addr.sin_family = AF_INET ;
    server_addr.sin_addr.s_addr = INADDR_ANY ;
    server_addr.sin_port = htons(SERVERPORT) ;

    check((bind(server_socket, (SA*)&server_addr , sizeof(server_addr))),
            "Bind Failed!") ;
    check(listen(server_socket, SERVER_BACKLOG),
            "Listen Failed!") ;
    
    while(true){
        printf("Waiting for connection...\n") ;
        addr_size = sizeof(SA_IN) ;
        check(client_socket = accept(server_socket , (SA*)&client_addr, (socklen_t*)&addr_size),
                "accept failed!!") ;
        printf("Connected\n") ;
    
        // pthread_t t ;
        int *pclient = (int *)malloc(sizeof(int)) ;
        *pclient = client_socket ;
        handle_connection(pclient) ;
        // this enqueue is causing the race condition.
        // pthread_mutex_lock(&lock) ;
        // enqueue(pclient) ;
        // pthread_mutex_unlock(&lock) ;
        // pthread_create(&t , NULL , handle_connection , (void *) pclient) ;
        

    }

    return 0 ;

}

int check(int exp, const char *msg) {
    if(exp == SOCKETERROR){
        perror(msg) ;
        exit(1) ;
    }
    return exp ;
}

// void * thread_function(void * args) {


void * handle_connection(void * p_clinet_soccket) {
    int clinet_soccket = *((int *) p_clinet_soccket) ;
    free(p_clinet_soccket) ;
    char buffer[BUFSIZE] ;
    size_t bytes_read ;
    int msgsize = 0 ;
    char actualpath[PATH_MAX + 1] ;

    while((bytes_read = read(clinet_soccket , buffer + msgsize , sizeof(buffer) - msgsize - 1)) > 0){
        msgsize += bytes_read ;
        if(msgsize > BUFSIZE - 1 || buffer[msgsize-1] == '\n') break ;
    }
    check(bytes_read , "recv error") ;
    buffer[msgsize - 1] = 0 ;

    printf("REQUEST: %s\n" , buffer) ;
    fflush(stdout) ;

    if(realpath(buffer , actualpath) == NULL){
        printf("ERROR(bad path): %s\n" , buffer) ;
        close(clinet_soccket) ;
        return NULL;
    }
    FILE *fp = fopen(actualpath , "r") ;

    if(fp == NULL){
        printf("ERROR(open): %s\n" , buffer) ;
        close(clinet_soccket) ;
        return NULL;
    }

    while((bytes_read = fread(buffer , 1 , BUFSIZE , fp)) > 0){
        printf("sending %zu bytes\n" , bytes_read) ;
        write(clinet_soccket , buffer , bytes_read) ;
    }
    close(clinet_soccket) ;
    fclose(fp) ;
    printf("CLosing connection\n") ;
    return NULL ;
}
#include <bits/stdc++.h>
#include <sys/socket.h>
#include <limits.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
// #include "myQueue.h"


#define SERVERPORT 8989
#define BUFSIZE 4096
#define SOCKETERROR (-1)
#define SERVER_BACKLOG 100
#define THREAD_POOL_SIZE 20

pthread_t thread_pool[THREAD_POOL_SIZE] ;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER ;
pthread_cond_t cond_var = PTHREAD_COND_INITIALIZER ;

typedef struct sockaddr_in SA_IN ;
typedef struct sockaddr SA ;

void * handle_connection(void * p_clinet_soccket) ;
int check(int exp , const char *msg) ;
void * thread_function(void * args) ;

int main(int argc , char **argv){
    int server_socket , client_socket, addr_size ;
    SA_IN server_addr , client_addr ;

    // for(int i =0 ; i < THREAD_POOL_SIZE ; i ++){
    //     pthread_create(&thread_pool[i] , NULL , thread_function , NULL) ;
    // }

    check((server_socket = socket(AF_INET , SOCK_STREAM , 0)),
            "Failed to create socket") ;
    
    server_addr.sin_family = AF_INET ;
    server_addr.sin_addr.s_addr = INADDR_ANY ;
    server_addr.sin_port = htons(SERVERPORT) ;

    check((bind(server_socket, (SA*)&server_addr , sizeof(server_addr))),
            "Bind Failed!") ;
    check(listen(server_socket, SERVER_BACKLOG),
            "Listen Failed!") ;
    
    while(true){
        printf("Waiting for connection...\n") ;
        addr_size = sizeof(SA_IN) ;
        check(client_socket = accept(server_socket , (SA*)&client_addr, (socklen_t*)&addr_size),
                "accept failed!!") ;
        printf("Connected\n") ;
    
        // pthread_t t ;
        int *pclient = (int *)malloc(sizeof(int)) ;
        *pclient = client_socket ;
        handle_connection(pclient) ;
        // this enqueue is causing the race condition.
        // pthread_mutex_lock(&lock) ;
        // enqueue(pclient) ;
        // pthread_mutex_unlock(&lock) ;
        // pthread_create(&t , NULL , handle_connection , (void *) pclient) ;
        

    }

    return 0 ;

}

int check(int exp, const char *msg) {
    if(exp == SOCKETERROR){
        perror(msg) ;
        exit(1) ;
    }
    return exp ;
}




void * handle_connection(void * p_clinet_soccket) {
    int clinet_soccket = *((int *) p_clinet_soccket) ;
    free(p_clinet_soccket) ;
    char buffer[BUFSIZE] ;
    size_t bytes_read ;
    int msgsize = 0 ;
    char actualpath[PATH_MAX + 1] ;

    while((bytes_read = read(clinet_soccket , buffer + msgsize , sizeof(buffer) - msgsize - 1)) > 0){
        msgsize += bytes_read ;
        if(msgsize > BUFSIZE - 1 || buffer[msgsize-1] == '\n') break ;
    }
    check(bytes_read , "recv error") ;
    buffer[msgsize - 1] = 0 ;

    printf("REQUEST: %s\n" , buffer) ;
    fflush(stdout) ;

    if(realpath(buffer , actualpath) == NULL){
        printf("ERROR(bad path): %s\n" , buffer) ;
        close(clinet_soccket) ;
        return NULL;
    }
    FILE *fp = fopen(actualpath , "r") ;

    if(fp == NULL){
        printf("ERROR(open): %s\n" , buffer) ;
        close(clinet_soccket) ;
        return NULL;
    }

    while((bytes_read = fread(buffer , 1 , BUFSIZE , fp)) > 0){
        printf("sending %zu bytes\n" , bytes_read) ;
        write(clinet_soccket , buffer , bytes_read) ;
    }
    close(clinet_soccket) ;
    fclose(fp) ;
    printf("CLosing connection\n") ;
    return NULL ;
}