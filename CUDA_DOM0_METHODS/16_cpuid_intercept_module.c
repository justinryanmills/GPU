/* CPUID intercept stub: clear ECX bit 19 (hypervisor). */
#include <linux/module.h>
#include <linux/kernel.h>

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("CPUID intercept to hide hypervisor bit for CUDA");
MODULE_AUTHOR("GPU_PROJECT");

static int __init cpuid_intercept_init(void)
{
	return 0;
}

static void __exit cpuid_intercept_exit(void)
{
}

module_init(cpuid_intercept_init);
module_exit(cpuid_intercept_exit);
