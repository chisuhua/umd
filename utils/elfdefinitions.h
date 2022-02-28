/*-
 * Copyright (c) 2010 Joseph Koshy
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * $Id: elfdefinitions.h 2064 2011-10-26 15:12:32Z jkoshy $
 */

/*
 * These definitions are based on:
 * - The public specification of the ELF format as defined in the
 *   October 2009 draft of System V ABI.
 *   See: http://www.sco.com/developers/gabi/latest/ch4.intro.html
 * - The May 1998 (version 1.5) draft of "The ELF-64 object format".
 * - Processor-specific ELF ABI definitions for sparc, i386, amd64, mips,
 *   ia64, and powerpc processors.
 * - The "Linkers and Libraries Guide", from Sun Microsystems.
 */

#ifndef _ELFDEFINITIONS_H_
#define _ELFDEFINITIONS_H_

#include "elfio/elf_types.hpp"

using namespace ELFIO;
/**
 ** ELF Types.
 **/

typedef uint32_t	Elf32_Addr;	/* Program address. */
typedef uint8_t		Elf32_Byte;	/* Unsigned tiny integer. */
typedef uint16_t	Elf32_Half;	/* Unsigned medium integer. */
typedef uint32_t	Elf32_Off;	/* File offset. */
typedef uint16_t	Elf32_Section;	/* Section index. */
typedef int32_t		Elf32_Sword;	/* Signed integer. */
typedef uint32_t	Elf32_Word;	/* Unsigned integer. */
typedef uint64_t	Elf32_Lword;	/* Unsigned long integer. */

typedef uint64_t	Elf64_Addr;	/* Program address. */
typedef uint8_t		Elf64_Byte;	/* Unsigned tiny integer. */
typedef uint16_t	Elf64_Half;	/* Unsigned medium integer. */
typedef uint64_t	Elf64_Off;	/* File offset. */
typedef uint16_t	Elf64_Section;	/* Section index. */
typedef int32_t		Elf64_Sword;	/* Signed integer. */
typedef uint32_t	Elf64_Word;	/* Unsigned integer. */
typedef uint64_t	Elf64_Lword;	/* Unsigned long integer. */
typedef uint64_t	Elf64_Xword;	/* Unsigned long integer. */
typedef int64_t		Elf64_Sxword;	/* Signed long integer. */


/*
 * Capability descriptors.
 */

/* 32-bit capability descriptor. */
typedef struct {
	Elf32_Word	c_tag;	     /* Type of entry. */
	union {
		Elf32_Word	c_val; /* Integer value. */
		Elf32_Addr	c_ptr; /* Pointer value. */
	} c_un;
} Elf32_Cap;

/* 64-bit capability descriptor. */
typedef struct {
	Elf64_Xword	c_tag;	     /* Type of entry. */
	union {
		Elf64_Xword	c_val; /* Integer value. */
		Elf64_Addr	c_ptr; /* Pointer value. */
	} c_un;
} Elf64_Cap;

/*
 * MIPS .conflict section entries.
 */

/* 32-bit entry. */
typedef struct {
	Elf32_Addr	c_index;
} Elf32_Conflict;

/* 64-bit entry. */
typedef struct {
	Elf64_Addr	c_index;
} Elf64_Conflict;

/*
 * Dynamic section entries.
 */
#if 0
/* 32-bit entry. */
typedef struct {
	Elf32_Sword	d_tag;	     /* Type of entry. */
	union {
		Elf32_Word	d_val; /* Integer value. */
		Elf32_Addr	d_ptr; /* Pointer value. */
	} d_un;
} Elf32_Dyn;

/* 64-bit entry. */
typedef struct {
	Elf64_Sxword	d_tag;	     /* Type of entry. */
	union {
		Elf64_Xword	d_val; /* Integer value. */
		Elf64_Addr	d_ptr; /* Pointer value; */
	} d_un;
} Elf64_Dyn;
#endif


/*
 * The executable header (EHDR).
 */
#if 0
/* 32 bit EHDR. */
typedef struct {
	unsigned char   e_ident[EI_NIDENT]; /* ELF identification. */
	Elf32_Half      e_type;	     /* Object file type (ET_*). */
	Elf32_Half      e_machine;   /* Machine type (EM_*). */
	Elf32_Word      e_version;   /* File format version (EV_*). */
	Elf32_Addr      e_entry;     /* Start address. */
	Elf32_Off       e_phoff;     /* File offset to the PHDR table. */
	Elf32_Off       e_shoff;     /* File offset to the SHDRheader. */
	Elf32_Word      e_flags;     /* Flags (EF_*). */
	Elf32_Half      e_ehsize;    /* Elf header size in bytes. */
	Elf32_Half      e_phentsize; /* PHDR table entry size in bytes. */
	Elf32_Half      e_phnum;     /* Number of PHDR entries. */
	Elf32_Half      e_shentsize; /* SHDR table entry size in bytes. */
	Elf32_Half      e_shnum;     /* Number of SHDR entries. */
	Elf32_Half      e_shstrndx;  /* Index of section name string table. */
} Elf32_Ehdr;


/* 64 bit EHDR. */
typedef struct {
	unsigned char   e_ident[EI_NIDENT]; /* ELF identification. */
	Elf64_Half      e_type;	     /* Object file type (ET_*). */
	Elf64_Half      e_machine;   /* Machine type (EM_*). */
	Elf64_Word      e_version;   /* File format version (EV_*). */
	Elf64_Addr      e_entry;     /* Start address. */
	Elf64_Off       e_phoff;     /* File offset to the PHDR table. */
	Elf64_Off       e_shoff;     /* File offset to the SHDRheader. */
	Elf64_Word      e_flags;     /* Flags (EF_*). */
	Elf64_Half      e_ehsize;    /* Elf header size in bytes. */
	Elf64_Half      e_phentsize; /* PHDR table entry size in bytes. */
	Elf64_Half      e_phnum;     /* Number of PHDR entries. */
	Elf64_Half      e_shentsize; /* SHDR table entry size in bytes. */
	Elf64_Half      e_shnum;     /* Number of SHDR entries. */
	Elf64_Half      e_shstrndx;  /* Index of section name string table. */
} Elf64_Ehdr;
#endif

typedef Elf64_Ehdr GElf_Ehdr;

/*
 * Shared object information.
 */

/* 32-bit entry. */
typedef struct {
	Elf32_Word l_name;	     /* The name of a shared object. */
	Elf32_Word l_time_stamp;     /* 32-bit timestamp. */
	Elf32_Word l_checksum;	     /* Checksum of visible symbols, sizes. */
	Elf32_Word l_version;	     /* Interface version string index. */
	Elf32_Word l_flags;	     /* Flags (LL_*). */
} Elf32_Lib;

/* 64-bit entry. */
typedef struct {
	Elf64_Word l_name;
	Elf64_Word l_time_stamp;
	Elf64_Word l_checksum;
	Elf64_Word l_version;
	Elf64_Word l_flags;
} Elf64_Lib;

#define	_ELF_DEFINE_LL_FLAGS()			\
_ELF_DEFINE_LL(LL_NONE,			0,	\
	"no flags")				\
_ELF_DEFINE_LL(LL_EXACT_MATCH,		0x1,	\
	"require an exact match")		\
_ELF_DEFINE_LL(LL_IGNORE_INT_VER,	0x2,	\
	"ignore version incompatibilities")	\
_ELF_DEFINE_LL(LL_REQUIRE_MINOR,	0x4,	\
	"")					\
_ELF_DEFINE_LL(LL_EXPORTS,		0x8,	\
	"")					\
_ELF_DEFINE_LL(LL_DELAY_LOAD,		0x10,	\
	"")					\
_ELF_DEFINE_LL(LL_DELTA,		0x20,	\
	"")

#undef	_ELF_DEFINE_LL
#define	_ELF_DEFINE_LL(N, V, DESCR)	N = V ,
enum {
	_ELF_DEFINE_LL_FLAGS()
	LL__LAST__
};

/*
 * Note tags
 */

#define	_ELF_DEFINE_NOTE_ENTRY_TYPES()					\
_ELF_DEFINE_NT(NT_ABI_TAG,	1,	"Tag indicating the ABI")	\
_ELF_DEFINE_NT(NT_GNU_HWCAP,	2,	"Hardware capabilities")	\
_ELF_DEFINE_NT(NT_GNU_BUILD_ID,	3,	"Build id, set by ld(1)")	\
_ELF_DEFINE_NT(NT_GNU_GOLD_VERSION, 4,					\
	"Version number of the GNU gold linker")			\
_ELF_DEFINE_NT(NT_PRSTATUS,	1,	"Process status")		\
_ELF_DEFINE_NT(NT_FPREGSET,	2,	"Floating point information")	\
_ELF_DEFINE_NT(NT_PRPSINFO,	3,	"Process information")		\
_ELF_DEFINE_NT(NT_AUXV,		6,	"Auxiliary vector")		\
_ELF_DEFINE_NT(NT_PRXFPREG,	0x46E62B7FUL,				\
	"Linux user_xfpregs structure")					\
_ELF_DEFINE_NT(NT_PSTATUS,	10,	"Linux process status")		\
_ELF_DEFINE_NT(NT_FPREGS,	12,	"Linux floating point regset")	\
_ELF_DEFINE_NT(NT_PSINFO,	13,	"Linux process information")	\
_ELF_DEFINE_NT(NT_LWPSTATUS,	16,	"Linux lwpstatus_t type")	\
_ELF_DEFINE_NT(NT_LWPSINFO,	17,	"Linux lwpinfo_t type")

#undef	_ELF_DEFINE_NT
#define	_ELF_DEFINE_NT(N, V, DESCR)	N = V ,
enum {
	_ELF_DEFINE_NOTE_ENTRY_TYPES()
	NT__LAST__
};

/* Aliases for the ABI tag. */
#define	NT_FREEBSD_ABI_TAG	NT_ABI_TAG
#define	NT_GNU_ABI_TAG		NT_ABI_TAG
#define	NT_NETBSD_IDENT		NT_ABI_TAG
#define	NT_OPENBSD_IDENT	NT_ABI_TAG

/*
 * Note descriptors.
 */

typedef	struct {
	uint32_t	n_namesz;    /* Length of note's name. */
	uint32_t	n_descsz;    /* Length of note's value. */
	uint32_t	n_type;	     /* Type of note. */
} Elf_Note;

typedef Elf_Note Elf32_Nhdr;	     /* 32-bit note header. */
typedef Elf_Note Elf64_Nhdr;	     /* 64-bit note header. */

/*
 * MIPS ELF options descriptor header.
 */

typedef struct {
	Elf64_Byte	kind;        /* Type of options. */
	Elf64_Byte     	size;	     /* Size of option descriptor. */
	Elf64_Half	section;     /* Index of section affected. */
	Elf64_Word	info;        /* Kind-specific information. */
} Elf_Options;

/*
 * Option kinds.
 */

#define	_ELF_DEFINE_OPTION_KINDS()					\
_ELF_DEFINE_ODK(ODK_NULL,       0,      "undefined")			\
_ELF_DEFINE_ODK(ODK_REGINFO,    1,      "register usage info")		\
_ELF_DEFINE_ODK(ODK_EXCEPTIONS, 2,      "exception processing info")	\
_ELF_DEFINE_ODK(ODK_PAD,        3,      "section padding")		\
_ELF_DEFINE_ODK(ODK_HWPATCH,    4,      "hardware patch applied")	\
_ELF_DEFINE_ODK(ODK_FILL,       5,      "fill value used by linker")	\
_ELF_DEFINE_ODK(ODK_TAGS,       6,      "reserved space for tools")	\
_ELF_DEFINE_ODK(ODK_HWAND,      7,      "hardware AND patch applied")	\
_ELF_DEFINE_ODK(ODK_HWOR,       8,      "hardware OR patch applied")	\
_ELF_DEFINE_ODK(ODK_GP_GROUP,   9,					\
	"GP group to use for text/data sections")			\
_ELF_DEFINE_ODK(ODK_IDENT,      10,     "ID information")		\
_ELF_DEFINE_ODK(ODK_PAGESIZE,   11,     "page size infomation")

#undef	_ELF_DEFINE_ODK
#define	_ELF_DEFINE_ODK(N, V, DESCR)	N = V ,
enum {
	_ELF_DEFINE_OPTION_KINDS()
	ODK__LAST__
};

/*
 * ODK_EXCEPTIONS info field masks.
 */

#define	_ELF_DEFINE_ODK_EXCEPTIONS_MASK()				\
_ELF_DEFINE_OEX(OEX_FPU_MIN,    0x0000001FUL,				\
	"minimum FPU exception which must be enabled")			\
_ELF_DEFINE_OEX(OEX_FPU_MAX,    0x00001F00UL,				\
	"maximum FPU exception which can be enabled")			\
_ELF_DEFINE_OEX(OEX_PAGE0,      0x00010000UL,				\
	"page zero must be mapped")					\
_ELF_DEFINE_OEX(OEX_SMM,        0x00020000UL,				\
	"run in sequential memory mode")				\
_ELF_DEFINE_OEX(OEX_PRECISEFP,  0x00040000UL,				\
	"run in precise FP exception mode")				\
_ELF_DEFINE_OEX(OEX_DISMISS,    0x00080000UL,				\
	"dismiss invalid address traps")

#undef	_ELF_DEFINE_OEX
#define	_ELF_DEFINE_OEX(N, V, DESCR)	N = V ,
enum {
	_ELF_DEFINE_ODK_EXCEPTIONS_MASK()
	OEX__LAST__
};

/*
 * ODK_PAD info field masks.
 */

#define	_ELF_DEFINE_ODK_PAD_MASK()					\
_ELF_DEFINE_OPAD(OPAD_PREFIX,   0x0001)					\
_ELF_DEFINE_OPAD(OPAD_POSTFIX,  0x0002)					\
_ELF_DEFINE_OPAD(OPAD_SYMBOL,   0x0004)

#undef	_ELF_DEFINE_OPAD
#define	_ELF_DEFINE_OPAD(N, V)		N = V ,
enum {
	_ELF_DEFINE_ODK_PAD_MASK()
	OPAD__LAST__
};

/*
 * ODK_HWPATCH info field masks.
 */

#define	_ELF_DEFINE_ODK_HWPATCH_MASK()					\
_ELF_DEFINE_OHW(OHW_R4KEOP,     0x00000001UL,				\
	"patch for R4000 branch at end-of-page bug")			\
_ELF_DEFINE_OHW(OHW_R8KPFETCH,  0x00000002UL,				\
	"R8000 prefetch bug may occur")					\
_ELF_DEFINE_OHW(OHW_R5KEOP,     0x00000004UL,				\
	"patch for R5000 branch at end-of-page bug")			\
_ELF_DEFINE_OHW(OHW_R5KCVTL,    0x00000008UL,				\
	"R5000 cvt.[ds].l bug: clean == 1")				\
_ELF_DEFINE_OHW(OHW_R10KLDL,    0x00000010UL,				\
	"needd patch for R10000 misaligned load")

#undef	_ELF_DEFINE_OHW
#define	_ELF_DEFINE_OHW(N, V, DESCR)	N = V ,
enum {
	_ELF_DEFINE_ODK_HWPATCH_MASK()
	OHW__LAST__
};

/*
 * ODK_HWAND/ODK_HWOR info field and hwp_flags[12] masks.
 */

#define	_ELF_DEFINE_ODK_HWP_MASK()					\
_ELF_DEFINE_HWP(OHWA0_R4KEOP_CHECKED, 0x00000001UL,			\
	"object checked for R4000 end-of-page bug")			\
_ELF_DEFINE_HWP(OHWA0_R4KEOP_CLEAN, 0x00000002UL,			\
	"object verified clean for R4000 end-of-page bug")		\
_ELF_DEFINE_HWP(OHWO0_FIXADE,   0x00000001UL,				\
	"object requires call to fixade")

#undef	_ELF_DEFINE_HWP
#define	_ELF_DEFINE_HWP(N, V, DESCR)	N = V ,
enum {
	_ELF_DEFINE_ODK_HWP_MASK()
	OHWX0__LAST__
};

/*
 * ODK_IDENT/ODK_GP_GROUP info field masks.
 */

#define	_ELF_DEFINE_ODK_GP_MASK()					\
_ELF_DEFINE_OGP(OGP_GROUP,      0x0000FFFFUL, "GP group number")	\
_ELF_DEFINE_OGP(OGP_SELF,       0x00010000UL,				\
	"GP group is self-contained")

#undef	_ELF_DEFINE_OGP
#define	_ELF_DEFINE_OGP(N, V, DESCR)	N = V ,
enum {
	_ELF_DEFINE_ODK_GP_MASK()
	OGP__LAST__
};

/*
 * MIPS ELF register info descriptor.
 */

/* 32 bit RegInfo entry. */
typedef struct {
	Elf32_Word	ri_gprmask;  /* Mask of general register used. */
	Elf32_Word	ri_cprmask[4]; /* Mask of coprocessor register used. */
	Elf32_Addr	ri_gp_value; /* GP register value. */
} Elf32_RegInfo;

/* 64 bit RegInfo entry. */
typedef struct {
	Elf64_Word	ri_gprmask;  /* Mask of general register used. */
	Elf64_Word	ri_pad;	     /* Padding. */
	Elf64_Word	ri_cprmask[4]; /* Mask of coprocessor register used. */
	Elf64_Addr	ri_gp_value; /* GP register value. */
} Elf64_RegInfo;

/*
 * Program Header Table (PHDR) entries.
 */
#if 0
/* 32 bit PHDR entry. */
typedef struct {
	Elf32_Word	p_type;	     /* Type of segment. */
	Elf32_Off	p_offset;    /* File offset to segment. */
	Elf32_Addr	p_vaddr;     /* Virtual address in memory. */
	Elf32_Addr	p_paddr;     /* Physical address (if relevant). */
	Elf32_Word	p_filesz;    /* Size of segment in file. */
	Elf32_Word	p_memsz;     /* Size of segment in memory. */
	Elf32_Word	p_flags;     /* Segment flags. */
	Elf32_Word	p_align;     /* Alignment constraints. */
} Elf32_Phdr;

/* 64 bit PHDR entry. */
typedef struct {
	Elf64_Word	p_type;	     /* Type of segment. */
	Elf64_Word	p_flags;     /* File offset to segment. */
	Elf64_Off	p_offset;    /* Virtual address in memory. */
	Elf64_Addr	p_vaddr;     /* Physical address (if relevant). */
	Elf64_Addr	p_paddr;     /* Size of segment in file. */
	Elf64_Xword	p_filesz;    /* Size of segment in memory. */
	Elf64_Xword	p_memsz;     /* Segment flags. */
	Elf64_Xword	p_align;     /* Alignment constraints. */
} Elf64_Phdr;
#endif

typedef Elf64_Phdr GElf_Phdr;

/*
 * Move entries, for describing data in COMMON blocks in a compact
 * manner.
 */

/* 32-bit move entry. */
typedef struct {
	Elf32_Lword	m_value;     /* Initialization value. */
	Elf32_Word 	m_info;	     /* Encoded size and index. */
	Elf32_Word	m_poffset;   /* Offset relative to symbol. */
	Elf32_Half	m_repeat;    /* Repeat count. */
	Elf32_Half	m_stride;    /* Number of units to skip. */
} Elf32_Move;

/* 64-bit move entry. */
typedef struct {
	Elf64_Lword	m_value;     /* Initialization value. */
	Elf64_Xword 	m_info;	     /* Encoded size and index. */
	Elf64_Xword	m_poffset;   /* Offset relative to symbol. */
	Elf64_Half	m_repeat;    /* Repeat count. */
	Elf64_Half	m_stride;    /* Number of units to skip. */
} Elf64_Move;
#ifndef ELF32_M_SYM
#define ELF32_M_SYM(I)		((I) >> 8)
#endif
#ifndef ELF32_M_SIZE
#define ELF32_M_SIZE(I)		((unsigned char) (I))
#endif
#ifndef ELF32_M_INFO
#define ELF32_M_INFO(M, S)	(((M) << 8) + (unsigned char) (S))
#endif

#ifndef ELF64_M_SYM
#define ELF64_M_SYM(I)		((I) >> 8)
#endif
#ifndef ELF64_M_SIZE
#define ELF64_M_SIZE(I)		((unsigned char) (I))
#endif
#ifndef ELF64_M_INFO
#define ELF64_M_INFO(M, S)	(((M) << 8) + (unsigned char) (S))
#endif

/*
 * Section Header Table (SHDR) entries.
 */
#if 0
/* 32 bit SHDR */
typedef struct {
	Elf32_Word	sh_name;     /* index of section name */
	Elf32_Word	sh_type;     /* section type */
	Elf32_Word	sh_flags;    /* section flags */
	Elf32_Addr	sh_addr;     /* in-memory address of section */
	Elf32_Off	sh_offset;   /* file offset of section */
	Elf32_Word	sh_size;     /* section size in bytes */
	Elf32_Word	sh_link;     /* section header table link */
	Elf32_Word	sh_info;     /* extra information */
	Elf32_Word	sh_addralign; /* alignment constraint */
	Elf32_Word	sh_entsize;   /* size for fixed-size entries */
} Elf32_Shdr;

/* 64 bit SHDR */
typedef struct {
	Elf64_Word	sh_name;     /* index of section name */
	Elf64_Word	sh_type;     /* section type */
	Elf64_Xword	sh_flags;    /* section flags */
	Elf64_Addr	sh_addr;     /* in-memory address of section */
	Elf64_Off	sh_offset;   /* file offset of section */
	Elf64_Xword	sh_size;     /* section size in bytes */
	Elf64_Word	sh_link;     /* section header table link */
	Elf64_Word	sh_info;     /* extra information */
	Elf64_Xword	sh_addralign; /* alignment constraint */
	Elf64_Xword	sh_entsize;  /* size for fixed-size entries */
} Elf64_Shdr;
#endif

typedef Elf64_Shdr GElf_Shdr;


/*
 * Symbol table entries.
 */
#if 0
typedef struct {
	Elf32_Word	st_name;     /* index of symbol's name */
	Elf32_Addr	st_value;    /* value for the symbol */
	Elf32_Word	st_size;     /* size of associated data */
	unsigned char	st_info;     /* type and binding attributes */
	unsigned char	st_other;    /* visibility */
	Elf32_Half	st_shndx;    /* index of related section */
} Elf32_Sym;

typedef struct {
	Elf64_Word	st_name;     /* index of symbol's name */
	unsigned char	st_info;     /* value for the symbol */
	unsigned char	st_other;    /* size of associated data */
	Elf64_Half	st_shndx;    /* type and binding attributes */
	Elf64_Addr	st_value;    /* visibility */
	Elf64_Xword	st_size;     /* index of related section */
} Elf64_Sym;
#endif

typedef Elf64_Sym GElf_Sym;

#ifndef ELF32_ST_BIND
#define ELF32_ST_BIND(I)	((I) >> 4)
#endif
#ifndef ELF32_ST_TYPE
#define ELF32_ST_TYPE(I)	((I) & 0xFU)
#endif
#ifndef ELF32_ST_INFO
#define ELF32_ST_INFO(B,T)	(((B) << 4) + ((T) & 0xF))
#endif

#ifndef ELF64_ST_BIND
#define ELF64_ST_BIND(I)	((I) >> 4)
#endif
#ifndef ELF64_ST_TYPE
#define ELF64_ST_TYPE(I)	((I) & 0xFU)
#endif
#ifndef ELF64_ST_INFO
#define ELF64_ST_INFO(B,T)	(((B) << 4) + ((T) & 0xF))
#endif

#ifndef ELF32_ST_VISIBILITY
#define ELF32_ST_VISIBILITY(O)	((O) & 0x3)
#endif

#ifndef ELF64_ST_VISIBILITY
#define ELF64_ST_VISIBILITY(O)	((O) & 0x3)
#endif

#define GELF_ST_TYPE(I) ELF64_ST_TYPE(I)
#define GELF_ST_BIND(I) ELF64_ST_BIND(I)
#define GELF_ST_INFO(B, T) ELF64_ST_INFO(B, T)

/*
 * Syminfo descriptors, containing additional symbol information.
 */

// typedef Elf64_Syminfo GElf_Syminfo;

typedef Elf64_Rela GElf_Rela;
typedef Elf64_Rel GElf_Rel;

#ifndef ELF32_R_SYM
#define ELF32_R_SYM(I)		((I) >> 8)
#endif
#ifndef ELF32_R_TYPE
#define ELF32_R_TYPE(I)		((unsigned char) (I))
#endif
#ifndef ELF32_R_INFO
#define ELF32_R_INFO(S,T)	(((S) << 8) + (unsigned char) (T))
#endif
#ifndef ELF64_R_SYM
#define ELF64_R_SYM(I)		((I) >> 32)
#endif
#ifndef ELF64_R_TYPE
#define ELF64_R_TYPE(I)		((I) & 0xFFFFFFFFUL)
#endif
#ifndef ELF64_R_INFO
#define ELF64_R_INFO(S,T)	(((S) << 32) + ((T) & 0xFFFFFFFFUL))
#endif

#define GELF_R_SYM(I) ELF64_R_SYM(I)
#define GELF_R_TYPE(I) ELF64_R_TYPE(I)
#define GELF_R_INFO(S,T) ELF64_R_INFO(S,T)
/*
 * Symbol versioning structures.
 */

typedef Elf32_Half	Elf32_Versym;

/* 64-bit structures. */

typedef struct {
	Elf64_Word	vda_name;    /* Index to name. */
	Elf64_Word	vda_next;    /* Offset to next entry. */
} Elf64_Verdaux;

typedef struct {
	Elf64_Word	vna_hash;    /* Hash value of dependency name. */
	Elf64_Half	vna_flags;   /* Flags. */
	Elf64_Half	vna_other;   /* Unused. */
	Elf64_Word	vna_name;    /* Offset to dependency name. */
	Elf64_Word	vna_next;    /* Offset to next vernaux entry. */
} Elf64_Vernaux;

typedef struct {
	Elf64_Half	vd_version;  /* Version information. */
	Elf64_Half	vd_flags;    /* Flags. */
	Elf64_Half	vd_ndx;	     /* Index into the versym section. */
	Elf64_Half	vd_cnt;	     /* Number of aux entries. */
	Elf64_Word	vd_hash;     /* Hash value of name. */
	Elf64_Word	vd_aux;	     /* Offset to aux entries. */
	Elf64_Word	vd_next;     /* Offset to next version definition. */
} Elf64_Verdef;

typedef struct {
	Elf64_Half	vn_version;  /* Version number. */
	Elf64_Half	vn_cnt;	     /* Number of aux entries. */
	Elf64_Word	vn_file;     /* Offset of associated file name. */
	Elf64_Word	vn_aux;	     /* Offset of vernaux array. */
	Elf64_Word	vn_next;     /* Offset of next verneed entry. */
} Elf64_Verneed;

typedef Elf64_Half	Elf64_Versym;


#ifndef	LIBELF_CONFIG_GNUHASH
#define	LIBELF_CONFIG_GNUHASH	1

/*
 * The header for GNU-style hash sections.
 */

typedef struct {
	uint32_t	gh_nbuckets;	/* Number of hash buckets. */
	uint32_t	gh_symndx;	/* First visible symbol in .dynsym. */
	uint32_t	gh_maskwords;	/* #maskwords used in bloom filter. */
	uint32_t	gh_shift2;	/* Bloom filter shift count. */
} Elf_GNU_Hash_Header;
#endif

#endif	/* _ELFDEFINITIONS_H_ */
