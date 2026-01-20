#define FUSE_USE_VERSION 26
#include <fuse.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

static const char *cpuinfo_path = "/tmp/cpuinfo_clean";

static int cpuinfo_getattr(const char *path, struct stat *stbuf)
{
    int res = 0;
    memset(stbuf, 0, sizeof(struct stat));
    
    if (strcmp(path, "/") == 0) {
        stbuf->st_mode = S_IFDIR | 0755;
        stbuf->st_nlink = 2;
    } else if (strcmp(path, "/cpuinfo") == 0) {
        stbuf->st_mode = S_IFREG | 0444;
        stbuf->st_nlink = 1;
        struct stat real_stat;
        if (stat(cpuinfo_path, &real_stat) == 0) {
            stbuf->st_size = real_stat.st_size;
        }
    } else {
        res = -ENOENT;
    }
    
    return res;
}

static int cpuinfo_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                           off_t offset, struct fuse_file_info *fi)
{
    (void) offset;
    (void) fi;
    
    if (strcmp(path, "/") != 0)
        return -ENOENT;
    
    filler(buf, ".", NULL, 0);
    filler(buf, "..", NULL, 0);
    filler(buf, "cpuinfo", NULL, 0);
    
    return 0;
}

static int cpuinfo_open(const char *path, struct fuse_file_info *fi)
{
    if (strcmp(path, "/cpuinfo") != 0)
        return -ENOENT;
    
    if ((fi->flags & O_ACCMODE) != O_RDONLY)
        return -EACCES;
    
    return 0;
}

static int cpuinfo_read(const char *path, char *buf, size_t size, off_t offset,
                        struct fuse_file_info *fi)
{
    (void) fi;
    
    if (strcmp(path, "/cpuinfo") != 0)
        return -ENOENT;
    
    FILE *fp = fopen(cpuinfo_path, "r");
    if (!fp)
        return -errno;
    
    fseek(fp, offset, SEEK_SET);
    size_t res = fread(buf, 1, size, fp);
    fclose(fp);
    
    return res;
}

static struct fuse_operations cpuinfo_oper = {
    .getattr = cpuinfo_getattr,
    .readdir = cpuinfo_readdir,
    .open = cpuinfo_open,
    .read = cpuinfo_read,
};

int main(int argc, char *argv[])
{
    return fuse_main(argc, argv, &cpuinfo_oper, NULL);
}

