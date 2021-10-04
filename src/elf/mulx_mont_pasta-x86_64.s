.text	

.globl	mulx_mont_pasta
.hidden	mulx_mont_pasta
.type	mulx_mont_pasta,@function
.align	32
mulx_mont_pasta:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa


	pushq	%rbp
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbp,-16
	pushq	%rbx
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbx,-24
	pushq	%r12
.cfi_adjust_cfa_offset	8
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_adjust_cfa_offset	8
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_adjust_cfa_offset	8
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_adjust_cfa_offset	8
.cfi_offset	%r15,-56
	subq	$8,%rsp
.cfi_adjust_cfa_offset	8


	movq	%rdx,%rbx
	movq	0(%rdx),%rdx
	movq	0(%rsi),%r14
	movq	8(%rsi),%r15
	movq	16(%rsi),%rbp
	movq	24(%rsi),%r9
	leaq	-128(%rsi),%rsi
	leaq	-128(%rcx),%rcx

	mulxq	%r14,%rax,%r11
	call	__mulx_mont_pasta

	movq	8(%rsp),%r15
.cfi_restore	%r15
	movq	16(%rsp),%r14
.cfi_restore	%r14
	movq	24(%rsp),%r13
.cfi_restore	%r13
	movq	32(%rsp),%r12
.cfi_restore	%r12
	movq	40(%rsp),%rbx
.cfi_restore	%rbx
	movq	48(%rsp),%rbp
.cfi_restore	%rbp
	leaq	56(%rsp),%rsp
.cfi_adjust_cfa_offset	-56

	.byte	0xf3,0xc3
.cfi_endproc	
.size	mulx_mont_pasta,.-mulx_mont_pasta

.globl	sqrx_mont_pasta
.hidden	sqrx_mont_pasta
.type	sqrx_mont_pasta,@function
.align	32
sqrx_mont_pasta:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa


	pushq	%rbp
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbp,-16
	pushq	%rbx
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbx,-24
	pushq	%r12
.cfi_adjust_cfa_offset	8
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_adjust_cfa_offset	8
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_adjust_cfa_offset	8
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_adjust_cfa_offset	8
.cfi_offset	%r15,-56
	subq	$8,%rsp
.cfi_adjust_cfa_offset	8


	movq	%rsi,%rbx
	movq	%rcx,%r8
	movq	%rdx,%rcx
	movq	0(%rsi),%rdx
	movq	8(%rsi),%r15
	movq	16(%rsi),%rbp
	movq	24(%rsi),%r9
	leaq	-128(%rbx),%rsi
	leaq	-128(%rcx),%rcx

	mulxq	%rdx,%rax,%r11
	call	__mulx_mont_pasta

	movq	8(%rsp),%r15
.cfi_restore	%r15
	movq	16(%rsp),%r14
.cfi_restore	%r14
	movq	24(%rsp),%r13
.cfi_restore	%r13
	movq	32(%rsp),%r12
.cfi_restore	%r12
	movq	40(%rsp),%rbx
.cfi_restore	%rbx
	movq	48(%rsp),%rbp
.cfi_restore	%rbp
	leaq	56(%rsp),%rsp
.cfi_adjust_cfa_offset	-56

	.byte	0xf3,0xc3
.cfi_endproc	
.size	sqrx_mont_pasta,.-sqrx_mont_pasta
.type	__mulx_mont_pasta,@function
.align	32
__mulx_mont_pasta:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa

	mulxq	%r15,%r15,%r12
	mulxq	%rbp,%rbp,%r13
	addq	%r15,%r11
	mulxq	%r9,%r9,%r14
	movq	8(%rbx),%rdx
	adcq	%rbp,%r12
	adcq	%r9,%r13
	adcq	$0,%r14

	movq	%rax,%r10
	imulq	%r8,%rax


	xorq	%r15,%r15
	mulxq	0+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r11
	adcxq	%r9,%r12

	mulxq	8+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r12
	adcxq	%r9,%r13

	mulxq	16+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r13
	adcxq	%r9,%r14

	mulxq	24+128(%rsi),%rbp,%r9
	movq	%rax,%rdx
	adoxq	%rbp,%r14
	adcxq	%r15,%r9
	adoxq	%r9,%r15


	mulxq	0+128(%rcx),%rbp,%rax
	adcxq	%rbp,%r10
	adoxq	%r11,%rax

	mulxq	8+128(%rcx),%rbp,%r9
	adcxq	%rbp,%rax
	adoxq	%r9,%r12

	mulxq	16+128(%rcx),%rbp,%r9
	adcxq	%rbp,%r12
	adoxq	%r9,%r13

	mulxq	24+128(%rcx),%rbp,%r9
	movq	16(%rbx),%rdx
	adcxq	%rbp,%r13
	adoxq	%r9,%r14
	adcxq	%r10,%r14
	adoxq	%r10,%r15
	adcxq	%r10,%r15
	movq	%rax,%r11
	imulq	%r8,%rax


	xorq	%r10,%r10
	mulxq	0+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r12
	adcxq	%r9,%r13

	mulxq	8+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r13
	adcxq	%r9,%r14

	mulxq	16+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r14
	adcxq	%r9,%r15

	mulxq	24+128(%rsi),%rbp,%r9
	movq	%rax,%rdx
	adoxq	%rbp,%r15
	adcxq	%r10,%r9
	adoxq	%r9,%r10


	mulxq	0+128(%rcx),%rbp,%rax
	adcxq	%rbp,%r11
	adoxq	%r12,%rax

	mulxq	8+128(%rcx),%rbp,%r9
	adcxq	%rbp,%rax
	adoxq	%r9,%r13

	mulxq	16+128(%rcx),%rbp,%r9
	adcxq	%rbp,%r13
	adoxq	%r9,%r14

	mulxq	24+128(%rcx),%rbp,%r9
	movq	24(%rbx),%rdx
	adcxq	%rbp,%r14
	adoxq	%r9,%r15
	adcxq	%r11,%r15
	adoxq	%r11,%r10
	adcxq	%r11,%r10
	movq	%rax,%r12
	imulq	%r8,%rax


	xorq	%r11,%r11
	mulxq	0+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r13
	adcxq	%r9,%r14

	mulxq	8+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r14
	adcxq	%r9,%r15

	mulxq	16+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r15
	adcxq	%r9,%r10

	mulxq	24+128(%rsi),%rbp,%r9
	movq	%rax,%rdx
	adoxq	%rbp,%r10
	adcxq	%r11,%r9
	adoxq	%r9,%r11


	mulxq	0+128(%rcx),%rbp,%rax
	adcxq	%rbp,%r12
	adoxq	%r13,%rax

	mulxq	8+128(%rcx),%rbp,%r9
	adcxq	%rbp,%rax
	adoxq	%r9,%r14

	mulxq	16+128(%rcx),%rbp,%r9
	adcxq	%rbp,%r14
	adoxq	%r9,%r15

	mulxq	24+128(%rcx),%rbp,%r9
	movq	%rax,%rdx
	adcxq	%rbp,%r15
	adoxq	%r9,%r10
	adcxq	%r12,%r10
	adoxq	%r12,%r11
	adcxq	%r12,%r11
	imulq	%r8,%rdx


	xorq	%r12,%r12
	mulxq	0+128(%rcx),%r13,%r9
	adcxq	%rax,%r13
	adoxq	%r9,%r14

	mulxq	8+128(%rcx),%rbp,%r9
	adcxq	%rbp,%r14
	adoxq	%r9,%r15

	mulxq	16+128(%rcx),%rbp,%r9
	adcxq	%rbp,%r15
	adoxq	%r9,%r10

	mulxq	24+128(%rcx),%rbp,%r9
	movq	%r14,%rdx
	leaq	128(%rcx),%rcx
	adcxq	%rbp,%r10
	adoxq	%r9,%r11
	movq	%r15,%rax
	adcxq	%r13,%r11
	adoxq	%r13,%r12
	adcq	$0,%r12




	movq	%r10,%rbp
	subq	0(%rcx),%r14
	sbbq	8(%rcx),%r15
	sbbq	16(%rcx),%r10
	movq	%r11,%r9
	sbbq	24(%rcx),%r11
	sbbq	$0,%r12

	cmovcq	%rdx,%r14
	cmovcq	%rax,%r15
	cmovcq	%rbp,%r10
	movq	%r14,0(%rdi)
	cmovcq	%r9,%r11
	movq	%r15,8(%rdi)
	movq	%r10,16(%rdi)
	movq	%r11,24(%rdi)

	.byte	0xf3,0xc3
.cfi_endproc
.size	__mulx_mont_pasta,.-__mulx_mont_pasta
.globl	mulx_mont_pasta
.hidden	mulx_mont_pasta
.type	mulx_mont_pasta,@function
.align	32
mulx_mont_pasta:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa


	pushq	%rbp
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbp,-16
	pushq	%rbx
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbx,-24
	pushq	%r12
.cfi_adjust_cfa_offset	8
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_adjust_cfa_offset	8
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_adjust_cfa_offset	8
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_adjust_cfa_offset	8
.cfi_offset	%r15,-56
	subq	$8,%rsp
.cfi_adjust_cfa_offset	8


	movq	%rdx,%rbx
	movq	0(%rdx),%rdx
	movq	0(%rsi),%r14
	movq	8(%rsi),%r15
	movq	16(%rsi),%rbp
	movq	24(%rsi),%r9
	leaq	-128(%rsi),%rsi
	leaq	-128(%rcx),%rcx

	mulxq	%r14,%rax,%r11
	call	__mulx_mont_pasta

	movq	8(%rsp),%r15
.cfi_restore	%r15
	movq	16(%rsp),%r14
.cfi_restore	%r14
	movq	24(%rsp),%r13
.cfi_restore	%r13
	movq	32(%rsp),%r12
.cfi_restore	%r12
	movq	40(%rsp),%rbx
.cfi_restore	%rbx
	movq	48(%rsp),%rbp
.cfi_restore	%rbp
	leaq	56(%rsp),%rsp
.cfi_adjust_cfa_offset	-56

	.byte	0xf3,0xc3
.cfi_endproc	
.size	mulx_mont_pasta,.-mulx_mont_pasta

.globl	sqrx_mont_pasta
.hidden	sqrx_mont_pasta
.type	sqrx_mont_pasta,@function
.align	32
sqrx_mont_pasta:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa


	pushq	%rbp
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbp,-16
	pushq	%rbx
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbx,-24
	pushq	%r12
.cfi_adjust_cfa_offset	8
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_adjust_cfa_offset	8
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_adjust_cfa_offset	8
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_adjust_cfa_offset	8
.cfi_offset	%r15,-56
	subq	$8,%rsp
.cfi_adjust_cfa_offset	8


	movq	%rsi,%rbx
	movq	%rcx,%r8
	movq	%rdx,%rcx
	movq	0(%rsi),%rdx
	movq	8(%rsi),%r15
	movq	16(%rsi),%rbp
	movq	24(%rsi),%r9
	leaq	-128(%rbx),%rsi
	leaq	-128(%rcx),%rcx

	mulxq	%rdx,%rax,%r11
	call	__mulx_mont_pasta

	movq	8(%rsp),%r15
.cfi_restore	%r15
	movq	16(%rsp),%r14
.cfi_restore	%r14
	movq	24(%rsp),%r13
.cfi_restore	%r13
	movq	32(%rsp),%r12
.cfi_restore	%r12
	movq	40(%rsp),%rbx
.cfi_restore	%rbx
	movq	48(%rsp),%rbp
.cfi_restore	%rbp
	leaq	56(%rsp),%rsp
.cfi_adjust_cfa_offset	-56

	.byte	0xf3,0xc3
.cfi_endproc	
.size	sqrx_mont_pasta,.-sqrx_mont_pasta
.type	__mulx_mont_pasta,@function
.align	32
__mulx_mont_pasta:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa

	mulxq	%r15,%r15,%r12
	mulxq	%rbp,%rbp,%r13
	addq	%r15,%r11
	mulxq	%r9,%r9,%r14
	movq	8(%rbx),%rdx
	adcq	%rbp,%r12
	adcq	%r9,%r13
	adcq	$0,%r14

	movq	%rax,%r10
	imulq	%r8,%rax


	xorq	%r15,%r15
	mulxq	0+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r11
	adcxq	%r9,%r12

	mulxq	8+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r12
	adcxq	%r9,%r13

	mulxq	16+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r13
	adcxq	%r9,%r14

	mulxq	24+128(%rsi),%rbp,%r9
	movq	%rax,%rdx
	adoxq	%rbp,%r14
	adcxq	%r15,%r9
	adoxq	%r9,%r15


	mulxq	0+128(%rcx),%rbp,%rax
	adcxq	%rbp,%r10
	adoxq	%r11,%rax

	mulxq	8+128(%rcx),%rbp,%r9
	adcxq	%rbp,%rax
	adoxq	%r9,%r12

	adcxq	%r10,%r12
	adoxq	%r10,%r13

	mulxq	24+128(%rcx),%rbp,%r9
	movq	16(%rbx),%rdx
	adcxq	%rbp,%r13
	adoxq	%r9,%r14
	adcxq	%r10,%r14
	adoxq	%r10,%r15
	adcxq	%r10,%r15
	movq	%rax,%r11
	imulq	%r8,%rax


	xorq	%r10,%r10
	mulxq	0+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r12
	adcxq	%r9,%r13

	mulxq	8+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r13
	adcxq	%r9,%r14

	mulxq	16+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r14
	adcxq	%r9,%r15

	mulxq	24+128(%rsi),%rbp,%r9
	movq	%rax,%rdx
	adoxq	%rbp,%r15
	adcxq	%r10,%r9
	adoxq	%r9,%r10


	mulxq	0+128(%rcx),%rbp,%rax
	adcxq	%rbp,%r11
	adoxq	%r12,%rax

	mulxq	8+128(%rcx),%rbp,%r9
	adcxq	%rbp,%rax
	adoxq	%r9,%r13

	adcxq	%r11,%r13
	adoxq	%r11,%r14

	mulxq	24+128(%rcx),%rbp,%r9
	movq	24(%rbx),%rdx
	adcxq	%rbp,%r14
	adoxq	%r9,%r15
	adcxq	%r11,%r15
	adoxq	%r11,%r10
	adcxq	%r11,%r10
	movq	%rax,%r12
	imulq	%r8,%rax


	xorq	%r11,%r11
	mulxq	0+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r13
	adcxq	%r9,%r14

	mulxq	8+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r14
	adcxq	%r9,%r15

	mulxq	16+128(%rsi),%rbp,%r9
	adoxq	%rbp,%r15
	adcxq	%r9,%r10

	mulxq	24+128(%rsi),%rbp,%r9
	movq	%rax,%rdx
	adoxq	%rbp,%r10
	adcxq	%r11,%r9
	adoxq	%r9,%r11


	mulxq	0+128(%rcx),%rbp,%rax
	adcxq	%rbp,%r12
	adoxq	%r13,%rax

	mulxq	8+128(%rcx),%rbp,%r9
	adcxq	%rbp,%rax
	adoxq	%r9,%r14

	adcxq	%r12,%r14
	adoxq	%r12,%r15

	mulxq	24+128(%rcx),%rbp,%r9
	movq	%rax,%rdx
	adcxq	%rbp,%r15
	adoxq	%r9,%r10
	adcxq	%r12,%r10
	adoxq	%r12,%r11
	adcxq	%r12,%r11
	imulq	%r8,%rdx


	xorq	%r12,%r12
	mulxq	0+128(%rcx),%r13,%r9
	adcxq	%rax,%r13
	adoxq	%r9,%r14

	mulxq	8+128(%rcx),%rbp,%r9
	adcxq	%rbp,%r14
	adoxq	%r9,%r15

	adcxq	%r13,%r15
	adoxq	%r13,%r10

	mulxq	24+128(%rcx),%rbp,%r9
	movq	%r14,%rdx
	leaq	128(%rcx),%rcx
	adcxq	%rbp,%r10
	adoxq	%r9,%r11
	movq	%r15,%rax
	adcxq	%r13,%r11
	adoxq	%r13,%r12
	adcq	$0,%r12




	movq	%r10,%rbp
	subq	0(%rcx),%r14
	sbbq	8(%rcx),%r15
	sbbq	16(%rcx),%r10
	movq	%r11,%r9
	sbbq	24(%rcx),%r11
	sbbq	$0,%r12

	cmovcq	%rdx,%r14
	cmovcq	%rax,%r15
	cmovcq	%rbp,%r10
	movq	%r14,0(%rdi)
	cmovcq	%r9,%r11
	movq	%r15,8(%rdi)
	movq	%r10,16(%rdi)
	movq	%r11,24(%rdi)

	.byte	0xf3,0xc3
.cfi_endproc
.size	__mulx_mont_pasta,.-__mulx_mont_pasta
.globl	fromx_mont_pasta
.hidden	fromx_mont_pasta
.type	fromx_mont_pasta,@function
.align	32
fromx_mont_pasta:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa


	pushq	%rbp
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbp,-16
	pushq	%rbx
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbx,-24
	pushq	%r12
.cfi_adjust_cfa_offset	8
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_adjust_cfa_offset	8
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_adjust_cfa_offset	8
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_adjust_cfa_offset	8
.cfi_offset	%r15,-56
	subq	$8,%rsp
.cfi_adjust_cfa_offset	8


	movq	%rdx,%rbx
	call	__mulx_by_1_mont_pasta





	movq	%r15,%rdx
	movq	%r10,%r12
	movq	%r11,%r13

	subq	0(%rbx),%r14
	sbbq	8(%rbx),%r15
	sbbq	16(%rbx),%r10
	sbbq	24(%rbx),%r11

	cmovncq	%r14,%rax
	cmovncq	%r15,%rdx
	cmovncq	%r10,%r12
	movq	%rax,0(%rdi)
	cmovncq	%r11,%r13
	movq	%rdx,8(%rdi)
	movq	%r12,16(%rdi)
	movq	%r13,24(%rdi)

	movq	8(%rsp),%r15
.cfi_restore	%r15
	movq	16(%rsp),%r14
.cfi_restore	%r14
	movq	24(%rsp),%r13
.cfi_restore	%r13
	movq	32(%rsp),%r12
.cfi_restore	%r12
	movq	40(%rsp),%rbx
.cfi_restore	%rbx
	movq	48(%rsp),%rbp
.cfi_restore	%rbp
	leaq	56(%rsp),%rsp
.cfi_adjust_cfa_offset	-56

	.byte	0xf3,0xc3
.cfi_endproc	
.size	fromx_mont_pasta,.-fromx_mont_pasta

.globl	redcx_mont_pasta
.hidden	redcx_mont_pasta
.type	redcx_mont_pasta,@function
.align	32
redcx_mont_pasta:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa


	pushq	%rbp
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbp,-16
	pushq	%rbx
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbx,-24
	pushq	%r12
.cfi_adjust_cfa_offset	8
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_adjust_cfa_offset	8
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_adjust_cfa_offset	8
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_adjust_cfa_offset	8
.cfi_offset	%r15,-56
	subq	$8,%rsp
.cfi_adjust_cfa_offset	8


	movq	%rdx,%rbx
	call	__mulx_by_1_mont_pasta

	addq	32(%rsi),%r14
	adcq	40(%rsi),%r15
	movq	%r14,%rax
	adcq	48(%rsi),%r10
	movq	%r15,%rdx
	adcq	56(%rsi),%r11
	sbbq	%rsi,%rsi




	movq	%r10,%r12
	subq	0(%rbx),%r14
	sbbq	8(%rbx),%r15
	sbbq	16(%rbx),%r10
	movq	%r11,%r13
	sbbq	24(%rbx),%r11
	sbbq	$0,%rsi

	cmovncq	%r14,%rax
	cmovncq	%r15,%rdx
	cmovncq	%r10,%r12
	movq	%rax,0(%rdi)
	cmovncq	%r11,%r13
	movq	%rdx,8(%rdi)
	movq	%r12,16(%rdi)
	movq	%r13,24(%rdi)

	movq	8(%rsp),%r15
.cfi_restore	%r15
	movq	16(%rsp),%r14
.cfi_restore	%r14
	movq	24(%rsp),%r13
.cfi_restore	%r13
	movq	32(%rsp),%r12
.cfi_restore	%r12
	movq	40(%rsp),%rbx
.cfi_restore	%rbx
	movq	48(%rsp),%rbp
.cfi_restore	%rbp
	leaq	56(%rsp),%rsp
.cfi_adjust_cfa_offset	-56

	.byte	0xf3,0xc3
.cfi_endproc	
.size	redcx_mont_pasta,.-redcx_mont_pasta
.type	__mulx_by_1_mont_pasta,@function
.align	32
__mulx_by_1_mont_pasta:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa

	movq	0(%rsi),%rax
	movq	8(%rsi),%r11
	movq	16(%rsi),%r12
	movq	24(%rsi),%r13

	movq	%rax,%r14
	imulq	%rcx,%rax
	movq	%rax,%r10

	mulq	0(%rbx)
	addq	%rax,%r14
	movq	%r10,%rax
	adcq	%rdx,%r14

	mulq	8(%rbx)
	addq	%rax,%r11
	movq	%r10,%rax
	adcq	$0,%rdx
	addq	%r14,%r11
	adcq	$0,%rdx
	movq	%rdx,%r14

	mulq	16(%rbx)
	movq	%r11,%r15
	imulq	%rcx,%r11
	addq	%rax,%r12
	movq	%r10,%rax
	adcq	$0,%rdx
	addq	%r14,%r12
	adcq	$0,%rdx
	movq	%rdx,%r14

	mulq	24(%rbx)
	addq	%rax,%r13
	movq	%r11,%rax
	adcq	$0,%rdx
	addq	%r14,%r13
	adcq	$0,%rdx
	movq	%rdx,%r14

	mulq	0(%rbx)
	addq	%rax,%r15
	movq	%r11,%rax
	adcq	%rdx,%r15

	mulq	8(%rbx)
	addq	%rax,%r12
	movq	%r11,%rax
	adcq	$0,%rdx
	addq	%r15,%r12
	adcq	$0,%rdx
	movq	%rdx,%r15

	mulq	16(%rbx)
	movq	%r12,%r10
	imulq	%rcx,%r12
	addq	%rax,%r13
	movq	%r11,%rax
	adcq	$0,%rdx
	addq	%r15,%r13
	adcq	$0,%rdx
	movq	%rdx,%r15

	mulq	24(%rbx)
	addq	%rax,%r14
	movq	%r12,%rax
	adcq	$0,%rdx
	addq	%r15,%r14
	adcq	$0,%rdx
	movq	%rdx,%r15

	mulq	0(%rbx)
	addq	%rax,%r10
	movq	%r12,%rax
	adcq	%rdx,%r10

	mulq	8(%rbx)
	addq	%rax,%r13
	movq	%r12,%rax
	adcq	$0,%rdx
	addq	%r10,%r13
	adcq	$0,%rdx
	movq	%rdx,%r10

	mulq	16(%rbx)
	movq	%r13,%r11
	imulq	%rcx,%r13
	addq	%rax,%r14
	movq	%r12,%rax
	adcq	$0,%rdx
	addq	%r10,%r14
	adcq	$0,%rdx
	movq	%rdx,%r10

	mulq	24(%rbx)
	addq	%rax,%r15
	movq	%r13,%rax
	adcq	$0,%rdx
	addq	%r10,%r15
	adcq	$0,%rdx
	movq	%rdx,%r10

	mulq	0(%rbx)
	addq	%rax,%r11
	movq	%r13,%rax
	adcq	%rdx,%r11

	mulq	8(%rbx)
	addq	%rax,%r14
	movq	%r13,%rax
	adcq	$0,%rdx
	addq	%r11,%r14
	adcq	$0,%rdx
	movq	%rdx,%r11

	mulq	16(%rbx)
	addq	%rax,%r15
	movq	%r13,%rax
	adcq	$0,%rdx
	addq	%r11,%r15
	adcq	$0,%rdx
	movq	%rdx,%r11

	mulq	24(%rbx)
	addq	%rax,%r10
	movq	%r14,%rax
	adcq	$0,%rdx
	addq	%r11,%r10
	adcq	$0,%rdx
	movq	%rdx,%r11
	.byte	0xf3,0xc3
.cfi_endproc
.size	__mulx_by_1_mont_pasta,.-__mulx_by_1_mont_pasta
.globl	sqrx_n_mul_mont_255
.hidden	sqrx_n_mul_mont_255
.type	sqrx_n_mul_mont_255,@function
.align	32
sqrx_n_mul_mont_255:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa


	pushq	%rbp
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbp,-16
	pushq	%rbx
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbx,-24
	pushq	%r12
.cfi_adjust_cfa_offset	8
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_adjust_cfa_offset	8
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_adjust_cfa_offset	8
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_adjust_cfa_offset	8
.cfi_offset	%r15,-56
	subq	$40,%rsp
.cfi_adjust_cfa_offset	40

	leaq	-128(%rcx),%rcx
	leaq	-128(%r8),%r8
	movq	%rcx,16(%rsp)
	movq	%rdi,24(%rsp)
	movq	%rdx,%rbx

	movq	0(%rsi),%rdx
	movq	8(%rsi),%rbp
	movq	16(%rsi),%rdi
	movq	24(%rsi),%rsi
	movq	%r9,(%rsp)
	movq	%rdx,%rcx
	jmp	.Loop_sqrx_255

.align	32
.Loop_sqrx_255:
	movq	%rbx,8(%rsp)

	xorq	%r15,%r15
	mulxq	%rdx,%rax,%r10
	mulxq	%rbp,%r11,%r9
	adcxq	%r10,%r11
	mulxq	%rdi,%r12,%rbx
	adcxq	%r9,%r12
	mulxq	%rsi,%r13,%r14
	movq	%rbp,%rdx
	adcxq	%rbx,%r13
	adcxq	%r15,%r14
	movq	%rax,%r10
	imulq	(%rsp),%rax


	xorq	%r15,%r15
	mulxq	%rcx,%r9,%rbx
	adoxq	%r9,%r11
	adcxq	%rbx,%r12

	mulxq	%rbp,%r9,%rbx
	adoxq	%r9,%r12
	adcxq	%rbx,%r13

	mulxq	%rdi,%r9,%rbx
	adoxq	%r9,%r13
	adcxq	%rbx,%r14

	mulxq	%rsi,%r9,%rbx
	movq	%rax,%rdx
	adoxq	%r9,%r14
	adcxq	%r15,%rbx
	adoxq	%rbx,%r15


	mulxq	0+128(%r8),%r9,%rax
	adcxq	%r9,%r10
	adoxq	%r11,%rax

	mulxq	8+128(%r8),%r9,%rbx
	adcxq	%r9,%rax
	adoxq	%rbx,%r12

	mulxq	16+128(%r8),%r9,%rbx
	adcxq	%r9,%r12
	adoxq	%rbx,%r13

	mulxq	24+128(%r8),%r9,%rbx
	movq	%rdi,%rdx
	adcxq	%r9,%r13
	adoxq	%rbx,%r14
	adcxq	%r10,%r14
	adoxq	%r10,%r15
	adcxq	%r10,%r15
	movq	%rax,%r11
	imulq	(%rsp),%rax


	xorq	%r10,%r10
	mulxq	%rcx,%r9,%rbx
	adoxq	%r9,%r12
	adcxq	%rbx,%r13

	mulxq	%rbp,%r9,%rbx
	adoxq	%r9,%r13
	adcxq	%rbx,%r14

	mulxq	%rdi,%r9,%rbx
	adoxq	%r9,%r14
	adcxq	%rbx,%r15

	mulxq	%rsi,%r9,%rbx
	movq	%rax,%rdx
	adoxq	%r9,%r15
	adcxq	%r10,%rbx
	adoxq	%rbx,%r10


	mulxq	0+128(%r8),%r9,%rax
	adcxq	%r9,%r11
	adoxq	%r12,%rax

	mulxq	8+128(%r8),%r9,%rbx
	adcxq	%r9,%rax
	adoxq	%rbx,%r13

	mulxq	16+128(%r8),%r9,%rbx
	adcxq	%r9,%r13
	adoxq	%rbx,%r14

	mulxq	24+128(%r8),%r9,%rbx
	movq	%rsi,%rdx
	adcxq	%r9,%r14
	adoxq	%rbx,%r15
	adcxq	%r11,%r15
	adoxq	%r11,%r10
	adcxq	%r11,%r10
	movq	%rax,%r12
	imulq	(%rsp),%rax


	xorq	%r11,%r11
	mulxq	%rcx,%r9,%rbx
	adoxq	%r9,%r13
	adcxq	%rbx,%r14

	mulxq	%rbp,%r9,%rbx
	adoxq	%r9,%r14
	adcxq	%rbx,%r15

	mulxq	%rdi,%r9,%rbx
	adoxq	%r9,%r15
	adcxq	%rbx,%r10

	mulxq	%rsi,%r9,%rbx
	movq	%rax,%rdx
	adoxq	%r9,%r10
	adcxq	%r11,%rbx
	adoxq	%rbx,%r11


	mulxq	0+128(%r8),%r9,%rax
	adcxq	%r9,%r12
	adoxq	%r13,%rax

	mulxq	8+128(%r8),%r9,%rbx
	adcxq	%r9,%rax
	adoxq	%rbx,%r14

	mulxq	16+128(%r8),%r9,%rbx
	adcxq	%r9,%r14
	adoxq	%rbx,%r15

	mulxq	24+128(%r8),%r9,%rbx
	movq	%rax,%rdx
	adcxq	%r9,%r15
	adoxq	%rbx,%r10
	adcxq	%r12,%r10
	adoxq	%r12,%r11
	adcxq	%r12,%r11
	imulq	(%rsp),%rdx
	movq	8(%rsp),%rbx
	xorq	%r12,%r12
	movq	16(%rsp),%r12


	mulxq	0+128(%r8),%r9,%rcx
	adcxq	%r9,%rax
	adoxq	%r14,%rcx

	mulxq	8+128(%r8),%r9,%rbp
	adcxq	%r9,%rcx
	adoxq	%r15,%rbp

	mulxq	16+128(%r8),%r9,%rdi
	adcxq	%r9,%rbp
	adoxq	%r10,%rdi

	mulxq	24+128(%r8),%r9,%rsi
	movq	%rcx,%rdx
	adcxq	%r9,%rdi
	adoxq	%r11,%rsi
	adcxq	%rax,%rsi

	subq	$1,%rbx
	jnz	.Loop_sqrx_255
	mulxq	0+128(%r12),%rax,%r13
	mulxq	8+128(%r12),%r14,%r9
	addq	%r13,%r14
	mulxq	16+128(%r12),%r15,%rbx
	adcq	%r9,%r15
	mulxq	24+128(%r12),%r10,%r11
	movq	%rbp,%rdx
	adcq	%rbx,%r10
	adcq	$0,%r11
	movq	%rax,%r13
	imulq	(%rsp),%rax


	xorq	%rcx,%rcx
	mulxq	0+128(%r12),%r9,%rbx
	adoxq	%r9,%r14
	adcxq	%rbx,%r15

	mulxq	8+128(%r12),%r9,%rbx
	adoxq	%r9,%r15
	adcxq	%rbx,%r10

	mulxq	16+128(%r12),%r9,%rbx
	adoxq	%r9,%r10
	adcxq	%rbx,%r11

	mulxq	24+128(%r12),%r9,%rbx
	movq	%rax,%rdx
	adoxq	%r9,%r11
	adcxq	%rcx,%rbx
	adoxq	%rbx,%rcx


	mulxq	0+128(%r8),%r9,%rax
	adcxq	%r9,%r13
	adoxq	%r14,%rax

	mulxq	8+128(%r8),%r9,%rbx
	adcxq	%r9,%rax
	adoxq	%rbx,%r15

	mulxq	16+128(%r8),%r9,%rbx
	adcxq	%r9,%r15
	adoxq	%rbx,%r10

	mulxq	24+128(%r8),%r9,%rbx
	movq	%rdi,%rdx
	adcxq	%r9,%r10
	adoxq	%rbx,%r11
	adcxq	%r13,%r11
	adoxq	%r13,%rcx
	adcxq	%r13,%rcx
	movq	%rax,%r14
	imulq	(%rsp),%rax


	xorq	%r13,%r13
	mulxq	0+128(%r12),%r9,%rbx
	adoxq	%r9,%r15
	adcxq	%rbx,%r10

	mulxq	8+128(%r12),%r9,%rbx
	adoxq	%r9,%r10
	adcxq	%rbx,%r11

	mulxq	16+128(%r12),%r9,%rbx
	adoxq	%r9,%r11
	adcxq	%rbx,%rcx

	mulxq	24+128(%r12),%r9,%rbx
	movq	%rax,%rdx
	adoxq	%r9,%rcx
	adcxq	%r13,%rbx
	adoxq	%rbx,%r13


	mulxq	0+128(%r8),%r9,%rax
	adcxq	%r9,%r14
	adoxq	%r15,%rax

	mulxq	8+128(%r8),%r9,%rbx
	adcxq	%r9,%rax
	adoxq	%rbx,%r10

	mulxq	16+128(%r8),%r9,%rbx
	adcxq	%r9,%r10
	adoxq	%rbx,%r11

	mulxq	24+128(%r8),%r9,%rbx
	movq	%rsi,%rdx
	adcxq	%r9,%r11
	adoxq	%rbx,%rcx
	adcxq	%r14,%rcx
	adoxq	%r14,%r13
	adcxq	%r14,%r13
	movq	%rax,%r15
	imulq	(%rsp),%rax


	xorq	%r14,%r14
	mulxq	0+128(%r12),%r9,%rbx
	adoxq	%r9,%r10
	adcxq	%rbx,%r11

	mulxq	8+128(%r12),%r9,%rbx
	adoxq	%r9,%r11
	adcxq	%rbx,%rcx

	mulxq	16+128(%r12),%r9,%rbx
	adoxq	%r9,%rcx
	adcxq	%rbx,%r13

	mulxq	24+128(%r12),%r9,%rbx
	movq	%rax,%rdx
	adoxq	%r9,%r13
	adcxq	%r14,%rbx
	adoxq	%rbx,%r14


	mulxq	0+128(%r8),%r9,%rax
	adcxq	%r9,%r15
	adoxq	%r10,%rax

	mulxq	8+128(%r8),%r9,%rbx
	adcxq	%r9,%rax
	adoxq	%rbx,%r11

	mulxq	16+128(%r8),%r9,%rbx
	adcxq	%r9,%r11
	adoxq	%rbx,%rcx

	mulxq	24+128(%r8),%r9,%rbx
	movq	%rax,%rdx
	adcxq	%r9,%rcx
	adoxq	%rbx,%r13
	adcxq	%r15,%r13
	adoxq	%r15,%r14
	adcxq	%r15,%r14
	imulq	(%rsp),%rdx
	movq	24(%rsp),%rdi


	xorq	%r15,%r15
	mulxq	0+128(%r8),%r9,%rbx
	adcxq	%r9,%rax
	adoxq	%rbx,%r11

	mulxq	8+128(%r8),%r9,%rbx
	adcxq	%r9,%r11
	adoxq	%rbx,%rcx

	mulxq	16+128(%r8),%r9,%rbx
	movq	%r11,%rax
	adcxq	%r9,%rcx
	adoxq	%rbx,%r13

	mulxq	24+128(%r8),%r9,%rbx
	leaq	128(%r8),%r8
	movq	%rcx,%rdx
	adcxq	%r9,%r13
	adoxq	%rbx,%r14
	adcxq	%r15,%r14




	movq	%r13,%r9
	subq	0(%r8),%r11
	sbbq	8(%r8),%rcx
	sbbq	16(%r8),%r13
	movq	%r14,%rbx
	sbbq	24(%r8),%r14

	cmovcq	%rax,%r11
	cmovcq	%rdx,%rcx
	cmovcq	%r9,%r13
	movq	%r11,0(%rdi)
	cmovcq	%rbx,%r14
	movq	%rcx,8(%rdi)
	movq	%r13,16(%rdi)
	movq	%r14,24(%rdi)

	movq	40(%rsp),%r15
.cfi_restore	%r15
	movq	48(%rsp),%r14
.cfi_restore	%r14
	movq	56(%rsp),%r13
.cfi_restore	%r13
	movq	64(%rsp),%r12
.cfi_restore	%r12
	movq	72(%rsp),%rbx
.cfi_restore	%rbx
	movq	80(%rsp),%rbp
.cfi_restore	%rbp
	leaq	88(%rsp),%rsp
.cfi_adjust_cfa_offset	-88

	.byte	0xf3,0xc3
.cfi_endproc	
.size	sqrx_n_mul_mont_255,.-sqrx_n_mul_mont_255
.globl	sqrx_n_mul_mont_pasta
.hidden	sqrx_n_mul_mont_pasta
.type	sqrx_n_mul_mont_pasta,@function
.align	32
sqrx_n_mul_mont_pasta:
.cfi_startproc
	.byte	0xf3,0x0f,0x1e,0xfa


	pushq	%rbp
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbp,-16
	pushq	%rbx
.cfi_adjust_cfa_offset	8
.cfi_offset	%rbx,-24
	pushq	%r12
.cfi_adjust_cfa_offset	8
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_adjust_cfa_offset	8
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_adjust_cfa_offset	8
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_adjust_cfa_offset	8
.cfi_offset	%r15,-56
	subq	$40,%rsp
.cfi_adjust_cfa_offset	40

	leaq	-128(%rcx),%rcx
	leaq	-128(%r8),%r8
	movq	%rcx,16(%rsp)
	movq	%rdi,24(%rsp)
	movq	%rdx,%rbx

	movq	0(%rsi),%rdx
	movq	8(%rsi),%rbp
	movq	16(%rsi),%rdi
	movq	24(%rsi),%rsi
	movq	%r9,(%rsp)
	movq	%rdx,%rcx
	jmp	.Loop_sqrx_pasta

.align	32
.Loop_sqrx_pasta:
	movq	%rbx,8(%rsp)

	mulxq	%rdx,%rax,%r10
	xorq	%r15,%r15
	mulxq	%rbp,%r11,%r9
	adcxq	%r10,%r11
	mulxq	%rdi,%r12,%rbx
	adcxq	%r9,%r12
	mulxq	%rsi,%r13,%r14
	movq	%rbp,%rdx
	adcxq	%rbx,%r13
	adcxq	%r15,%r14
	movq	%rax,%r10
	imulq	(%rsp),%rax


	xorq	%r15,%r15
	mulxq	%rcx,%r9,%rbx
	adoxq	%r9,%r11
	adcxq	%rbx,%r12

	mulxq	%rbp,%r9,%rbx
	adoxq	%r9,%r12
	adcxq	%rbx,%r13

	mulxq	%rdi,%r9,%rbx
	adoxq	%r9,%r13
	adcxq	%rbx,%r14

	mulxq	%rsi,%r9,%rbx
	movq	%rax,%rdx
	adoxq	%r9,%r14
	adcxq	%r15,%rbx
	adoxq	%rbx,%r15


	mulxq	0+128(%r8),%r9,%rax
	xorq	%rbx,%rbx
	adcxq	%r9,%r10
	adoxq	%r11,%rax

	mulxq	8+128(%r8),%r9,%rbx
	adcxq	%r9,%rax
	adoxq	%rbx,%r12

	adcxq	%r10,%r12
	adoxq	%r10,%r13

	mulxq	24+128(%r8),%r9,%rbx
	movq	%rdi,%rdx
	adcxq	%r9,%r13
	adoxq	%rbx,%r14
	adcxq	%r10,%r14
	adoxq	%r10,%r15
	adcxq	%r10,%r15
	movq	%rax,%r11
	imulq	(%rsp),%rax


	xorq	%r10,%r10
	mulxq	%rcx,%r9,%rbx
	adoxq	%r9,%r12
	adcxq	%rbx,%r13

	mulxq	%rbp,%r9,%rbx
	adoxq	%r9,%r13
	adcxq	%rbx,%r14

	mulxq	%rdi,%r9,%rbx
	adoxq	%r9,%r14
	adcxq	%rbx,%r15

	mulxq	%rsi,%r9,%rbx
	movq	%rax,%rdx
	adoxq	%r9,%r15
	adcxq	%r10,%rbx
	adoxq	%rbx,%r10


	mulxq	0+128(%r8),%r9,%rax
	xorq	%rbx,%rbx
	adcxq	%r9,%r11
	adoxq	%r12,%rax

	mulxq	8+128(%r8),%r9,%rbx
	adcxq	%r9,%rax
	adoxq	%rbx,%r13

	adcxq	%r11,%r13
	adoxq	%r11,%r14

	mulxq	24+128(%r8),%r9,%rbx
	movq	%rsi,%rdx
	adcxq	%r9,%r14
	adoxq	%rbx,%r15
	adcxq	%r11,%r15
	adoxq	%r11,%r10
	adcxq	%r11,%r10
	movq	%rax,%r12
	imulq	(%rsp),%rax


	xorq	%r11,%r11
	mulxq	%rcx,%r9,%rbx
	adoxq	%r9,%r13
	adcxq	%rbx,%r14

	mulxq	%rbp,%r9,%rbx
	adoxq	%r9,%r14
	adcxq	%rbx,%r15

	mulxq	%rdi,%r9,%rbx
	adoxq	%r9,%r15
	adcxq	%rbx,%r10

	mulxq	%rsi,%r9,%rbx
	movq	%rax,%rdx
	adoxq	%r9,%r10
	adcxq	%r11,%rbx
	adoxq	%rbx,%r11


	mulxq	0+128(%r8),%r9,%rax
	xorq	%rbx,%rbx
	adcxq	%r9,%r12
	adoxq	%r13,%rax

	mulxq	8+128(%r8),%r9,%rbx
	adcxq	%r9,%rax
	adoxq	%rbx,%r14

	adcxq	%r12,%r14
	adoxq	%r12,%r15

	mulxq	24+128(%r8),%r9,%rbx
	movq	%rax,%rdx
	adcxq	%r9,%r15
	adoxq	%rbx,%r10
	adcxq	%r12,%r10
	adoxq	%r12,%r11
	adcxq	%r12,%r11
	imulq	(%rsp),%rdx
	movq	8(%rsp),%rbx
	movq	16(%rsp),%r12


	xorq	%rdi,%rdi
	mulxq	0+128(%r8),%r9,%rcx
	adcxq	%r9,%rax
	adoxq	%r14,%rcx

	mulxq	8+128(%r8),%r9,%rbp
	adcxq	%r9,%rcx
	adoxq	%r15,%rbp

	adcxq	%rax,%rbp
	adoxq	%r10,%rdi

	mulxq	24+128(%r8),%r9,%rsi
	movq	%rcx,%rdx
	adcxq	%r9,%rdi
	adoxq	%r11,%rsi
	adcxq	%rax,%rsi

	subq	$1,%rbx
	jnz	.Loop_sqrx_pasta
	mulxq	0+128(%r12),%rax,%r13
	xorq	%rcx,%rcx
	mulxq	8+128(%r12),%r14,%r9
	adcxq	%r13,%r14
	mulxq	16+128(%r12),%r15,%rbx
	adcxq	%r9,%r15
	mulxq	24+128(%r12),%r10,%r11
	movq	%rbp,%rdx
	adcxq	%rbx,%r10
	adcxq	%rcx,%r11
	movq	%rax,%r13
	imulq	(%rsp),%rax


	xorq	%rcx,%rcx
	mulxq	0+128(%r12),%r9,%rbx
	adoxq	%r9,%r14
	adcxq	%rbx,%r15

	mulxq	8+128(%r12),%r9,%rbx
	adoxq	%r9,%r15
	adcxq	%rbx,%r10

	mulxq	16+128(%r12),%r9,%rbx
	adoxq	%r9,%r10
	adcxq	%rbx,%r11

	mulxq	24+128(%r12),%r9,%rbx
	movq	%rax,%rdx
	adoxq	%r9,%r11
	adcxq	%rcx,%rbx
	adoxq	%rbx,%rcx


	mulxq	0+128(%r8),%r9,%rax
	adcxq	%r9,%r13
	adoxq	%r14,%rax

	mulxq	8+128(%r8),%r9,%rbx
	adcxq	%r9,%rax
	adoxq	%rbx,%r15

	adcxq	%r13,%r15
	adoxq	%r13,%r10

	mulxq	24+128(%r8),%r9,%rbx
	movq	%rdi,%rdx
	adcxq	%r9,%r10
	adoxq	%rbx,%r11
	adcxq	%r13,%r11
	adoxq	%r13,%rcx
	adcxq	%r13,%rcx
	movq	%rax,%r14
	imulq	(%rsp),%rax


	xorq	%r13,%r13
	mulxq	0+128(%r12),%r9,%rbx
	adoxq	%r9,%r15
	adcxq	%rbx,%r10

	mulxq	8+128(%r12),%r9,%rbx
	adoxq	%r9,%r10
	adcxq	%rbx,%r11

	mulxq	16+128(%r12),%r9,%rbx
	adoxq	%r9,%r11
	adcxq	%rbx,%rcx

	mulxq	24+128(%r12),%r9,%rbx
	movq	%rax,%rdx
	adoxq	%r9,%rcx
	adcxq	%r13,%rbx
	adoxq	%rbx,%r13


	mulxq	0+128(%r8),%r9,%rax
	adcxq	%r9,%r14
	adoxq	%r15,%rax

	mulxq	8+128(%r8),%r9,%rbx
	adcxq	%r9,%rax
	adoxq	%rbx,%r10

	adcxq	%r14,%r10
	adoxq	%r14,%r11

	mulxq	24+128(%r8),%r9,%rbx
	movq	%rsi,%rdx
	adcxq	%r9,%r11
	adoxq	%rbx,%rcx
	adcxq	%r14,%rcx
	adoxq	%r14,%r13
	adcxq	%r14,%r13
	movq	%rax,%r15
	imulq	(%rsp),%rax


	xorq	%r14,%r14
	mulxq	0+128(%r12),%r9,%rbx
	adoxq	%r9,%r10
	adcxq	%rbx,%r11

	mulxq	8+128(%r12),%r9,%rbx
	adoxq	%r9,%r11
	adcxq	%rbx,%rcx

	mulxq	16+128(%r12),%r9,%rbx
	adoxq	%r9,%rcx
	adcxq	%rbx,%r13

	mulxq	24+128(%r12),%r9,%rbx
	movq	%rax,%rdx
	adoxq	%r9,%r13
	adcxq	%r14,%rbx
	adoxq	%rbx,%r14


	mulxq	0+128(%r8),%r9,%rax
	adcxq	%r9,%r15
	adoxq	%r10,%rax

	mulxq	8+128(%r8),%r9,%rbx
	adcxq	%r9,%rax
	adoxq	%rbx,%r11

	adcxq	%r15,%r11
	adoxq	%r15,%rcx

	mulxq	24+128(%r8),%r9,%rbx
	movq	%rax,%rdx
	adcxq	%r9,%rcx
	adoxq	%rbx,%r13
	adcxq	%r15,%r13
	adoxq	%r15,%r14
	adcxq	%r15,%r14
	imulq	(%rsp),%rdx
	movq	24(%rsp),%rdi


	xorq	%r15,%r15
	mulxq	0+128(%r8),%r9,%rbx
	adcxq	%r9,%rax
	adoxq	%rbx,%r11

	mulxq	8+128(%r8),%r9,%rbx
	adcxq	%r9,%r11
	adoxq	%rbx,%rcx


	adcxq	%r15,%rcx
	adoxq	%r15,%r13

	mulxq	24+128(%r8),%r9,%rbx


	adcxq	%r9,%r13
	adoxq	%rbx,%r14
	adcxq	%r15,%r14














	movq	%r11,0(%rdi)

	movq	%rcx,8(%rdi)
	movq	%r13,16(%rdi)
	movq	%r14,24(%rdi)

	movq	40(%rsp),%r15
.cfi_restore	%r15
	movq	48(%rsp),%r14
.cfi_restore	%r14
	movq	56(%rsp),%r13
.cfi_restore	%r13
	movq	64(%rsp),%r12
.cfi_restore	%r12
	movq	72(%rsp),%rbx
.cfi_restore	%rbx
	movq	80(%rsp),%rbp
.cfi_restore	%rbp
	leaq	88(%rsp),%rsp
.cfi_adjust_cfa_offset	-88

	.byte	0xf3,0xc3
.cfi_endproc	
.size	sqrx_n_mul_mont_pasta,.-sqrx_n_mul_mont_pasta

.section	.note.gnu.property,"a",@note
	.long	4,2f-1f,5
	.byte	0x47,0x4E,0x55,0
1:	.long	0xc0000002,4,3
.align	8
2:
