#include "qemu/osdep.h"
#include "hw/pci/pci.h"
#include "hw/qdev-properties.h"
#include "migration/vmstate.h"
#include "qemu/module.h"
#include "qapi/error.h"

/* Device type name */
#define TYPE_VGPU_STUB "vgpu-stub"

/* Object cast macro */
#define VGPU_STUB(obj) \
    OBJECT_CHECK(VGPUStubState, (obj), TYPE_VGPU_STUB)

/* Device state structure */
typedef struct VGPUStubState {
    PCIDevice parent_obj;      /* Must be first */
    
    /* Memory region for BAR0 (MMIO) */
    MemoryRegion mmio;
    
    /* BAR size (4KB minimum for Step 1) */
    uint32_t bar_size;
    
    /* Flag to track BAR sizing operation */
    bool bar_size_set;
    
    /* Communication registers (minimal for Step 1) */
    uint32_t command_reg;      /* Offset 0x000 in BAR */
    uint32_t status_reg;        /* Offset 0x004 in BAR */
    
} VGPUStubState;

/* MMIO read handler - called when VM reads from BAR */
static uint64_t vgpu_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    VGPUStubState *s = opaque;
    uint64_t val = 0;
    
    /* For Step 1, minimal implementation - just return register values */
    switch (addr) {
    case 0x000:  /* Command register */
        val = s->command_reg;
        break;
    case 0x004:  /* Status register */
        val = s->status_reg;
        break;
    default:
        /* Unmapped addresses return 0 */
        val = 0;
        break;
    }
    
    return val;
}

/* MMIO write handler - called when VM writes to BAR */
static void vgpu_mmio_write(void *opaque, hwaddr addr,
                            uint64_t val, unsigned size)
{
    VGPUStubState *s = opaque;
    
    /* For Step 1, minimal implementation - just store writes */
    switch (addr) {
    case 0x000:  /* Command register */
        s->command_reg = val;
        /* TODO: In later steps, trigger host-side processing */
        break;
    default:
        /* Ignore writes to unmapped addresses */
        break;
    }
}

/* Memory region operations - defines how MMIO behaves */
static const MemoryRegionOps vgpu_mmio_ops = {
    .read = vgpu_mmio_read,
    .write = vgpu_mmio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 4,  /* 32-bit accesses */
        .max_access_size = 4,
    },
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* Device realization - called when device is created */
static void vgpu_realize(PCIDevice *pci_dev, Error **errp)
{
    VGPUStubState *s = VGPU_STUB(pci_dev);
    uint8_t *pci_conf = pci_dev->config;
    
    /* ============================================================
     * PCI CONFIGURATION SPACE SETUP
     * ============================================================
     * These fields are REQUIRED for Linux to enumerate the device
     */
    
    /* Vendor ID: 0x1AF4 (Red Hat - commonly used for virtual devices) */
    pci_set_word(pci_conf + PCI_VENDOR_ID, 0x1AF4);
    
    /* Device ID: 0x1111 (Custom - can be changed if conflicts) */
    pci_set_word(pci_conf + PCI_DEVICE_ID, 0x1111);
    
    /* Command register: Initially disabled (0x0000)
     * Bits: I/O space enable, memory space enable, etc.
     * Linux will enable these during enumeration */
    pci_set_word(pci_conf + PCI_COMMAND, 0x0000);
    
    /* Status register: Indicate capabilities list present
     * Bit 4 = Capabilities List (0x0010)
     * For Step 1, we don't have capabilities, but setting this
     * bit is safe and improves compatibility */
    pci_set_word(pci_conf + PCI_STATUS, PCI_STATUS_CAP_LIST);
    
    /* Revision ID: 0x01 (first version) */
    pci_set_byte(pci_conf + PCI_REVISION_ID, 0x01);
    
    /* Class Code: Processing Accelerator, General Purpose
     * Byte 0 (0x09): Base class = 0x12 (Processing Accelerator)
     * Byte 1 (0x0A): Subclass = 0x00 (General Purpose)
     * Byte 2 (0x0B): Programming interface = 0x00 (Generic)
     * 
     * NOTE: Using Processing Accelerator instead of VGA because:
     * - This is a compute-focused vGPU (CUDA workloads)
     * - No display functionality
     * - Avoids VGA/DRM driver conflicts
     * - Semantically correct for compute devices */
    pci_set_byte(pci_conf + PCI_CLASS_DEVICE + 0, 0x12);
    pci_set_byte(pci_conf + PCI_CLASS_DEVICE + 1, 0x00);
    pci_set_byte(pci_conf + PCI_CLASS_DEVICE + 2, 0x00);
    
    /* Header Type: 0x00 = Standard PCI device (single function) */
    pci_set_byte(pci_conf + PCI_HEADER_TYPE, PCI_HEADER_TYPE_NORMAL);
    
    /* Subsystem Vendor ID: Same as vendor ID */
    pci_set_word(pci_conf + PCI_SUBSYSTEM_VENDOR_ID, 0x1AF4);
    
    /* Subsystem Device ID: Same as device ID */
    pci_set_word(pci_conf + PCI_SUBSYSTEM_ID, 0x1111);
    
    /* Capabilities pointer: 0x00 = No capabilities for Step 1 */
    pci_set_byte(pci_conf + PCI_CAPABILITY_LIST, 0x00);
    
    /* Interrupt Line: 0xFF = Not connected (we'll use MSI/MSI-X later) */
    pci_set_byte(pci_conf + PCI_INTERRUPT_LINE, 0xFF);
    
    /* Interrupt Pin: 0x00 = No INTx pin used */
    pci_set_byte(pci_conf + PCI_INTERRUPT_PIN, 0x00);
    
    /* ============================================================
     * BAR (BASE ADDRESS REGISTER) SETUP
     * ============================================================
     * BAR0 will be a 4KB memory-mapped I/O region
     */
    
    s->bar_size = 0x1000;  /* 4KB = 4096 bytes */
    s->bar_size_set = false;
    
    /* Initialize memory region for BAR0 */
    memory_region_init_io(&s->mmio, OBJECT(s), &vgpu_mmio_ops, s,
                          "vgpu-stub-mmio", s->bar_size);
    
    /* Register BAR0 as a memory-mapped I/O region
     * PCI_BASE_ADDRESS_SPACE_MEMORY = Memory space (not I/O space)
     * This makes the BAR appear as a memory region to the VM */
    pci_register_bar(pci_dev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);
    
    /* Initialize device registers */
    s->command_reg = 0;
    s->status_reg = 0;
    
    /* Device is now ready! */
}

/* Device class initialization */
static void vgpu_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);
    
    /* Set device category - use MISC instead of DISPLAY for compute device */
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
    
    /* Device description */
    dc->desc = "Virtual GPU Stub Device (Step 1 - Compute Accelerator)";
    
    /* PCI device class setup */
    k->realize = vgpu_realize;  /* Called when device is created */
    k->vendor_id = 0x1AF4;       /* Red Hat */
    k->device_id = 0x1111;       /* Custom device ID */
    k->class_id = 0x120000;      /* Processing Accelerator (0x12 base class) */
    
    /* Enable hotplug support (optional, for future use) */
    k->hotpluggable = false;
}

/* Device instance initialization */
static void vgpu_instance_init(Object *obj)
{
    /* Nothing to initialize here for Step 1 */
    /* Device state is initialized in realize() */
}

/* Type information structure */
static const TypeInfo vgpu_stub_info = {
    .name = TYPE_VGPU_STUB,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(VGPUStubState),
    .instance_init = vgpu_instance_init,
    .class_init = vgpu_class_init,
    .interfaces = (InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

/* Register device type */
static void vgpu_register_types(void)
{
    type_register_static(&vgpu_stub_info);
}

/* QEMU module initialization */
type_init(vgpu_register_types);


