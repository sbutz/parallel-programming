#include <stdio.h>
#include <string.h> 
#define MAXCHAR 1024

int read_file(const char* filename, unsigned char** data) {
    FILE *fp;
    char str[MAXCHAR];
    size_t read;
    size_t total_size=0; 
    *data=NULL;
    fp = fopen(filename, "r");
    if (fp == NULL){
        printf("Could not open file %s\n",filename);
        return 1;
    }
    while (fgets(str, MAXCHAR, fp) != NULL) {
	read=strlen(str);
	*data=(unsigned char*)realloc(*data, (total_size+read)*sizeof(char));
	memcpy(*data+total_size, str, read*sizeof(char));
	total_size+=read;
    }
    fclose(fp);
    return total_size;
}
