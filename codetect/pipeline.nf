#!/usr/bin/env nextflow

samples = Channel.fromFilePairs(params.readsFolder+"/*_R{1,2}.fastq.gz")

process pickReference{ 
    publishDir params.outDir + "/mapping"

    input: 
        tuple val(sample_id), file(reads) from samples
    output:
        file "${opref}.fa" into ref
        tuple val(sample_id), file(reads) into samples2
    script:
        opref = sample_id
        vapordb = file(params.vapordb)
        println "Choosing a ref"
    """
    vapor.py --return_seqs -fa ${vapordb} -fq ${reads} > ${opref}.fa
    """
}

process mapReads {
    publishDir params.outDir+"/mapping"

    input: 
        file vpref from ref
        tuple val(sample_id), file(reads) from samples2
    output:
        tuple file("${opref}.bam"), file("${opref}.bai") into bamFileAndIndex
        file vpref into ref2
    script:
        opref = sample_id
    """
    minimap2 -ax sr ${vpref} ${reads} | samtools view -b | samtools sort > ${opref}.bam
    samtools index ${opref}.bam ${opref}.bai
    """
}

process runEM {
    publishDir = params.outDir+"/estimation"

    input:
        tuple file(bamFile), file(bamIndex) from bamFileAndIndex
        file vpref from ref2
    output:
        tuple file(bamFile), file(bamIndex) into bamFileAndIndex2
        file vpref into ref3
        file "${opref}.emout" into EMOutFile
    script:
        opref = bamFile.name.replace(".bam", "")
        mind = params.min_d        
    """
    codetectem.py -mind ${mind} -bam ${bamFile} -ref ${vpref} > ${opref}.emout
    """    
}

//process runGibbs {
//   publishDir = params.outDir+"/estimation"
//
//    input:
//        tuple file(bamFile), file(bamIndex) from bamFileAndIndex2
//        file vpref from ref3
//    output:
//        tuple file(bamFile), file(bamIndex) into bamFileAndIndex3
//        file vpref into ref4
//        file "${opref}.gibbsout" into GibbsOutFile
//    script:
//        opref = bamFile.name.replace(".bam", "")
//        mind = params.min_d
//    """
//    codetectgibbs.py -mind ${mind} -bam ${bamFile} -ref ${vpref} > ${opref}.gibbsout
//    """
//}

process runMH{
    publishDir = params.outDir+"/estimation"

    input:
        tuple file(bamFile), file(bamIndex) from bamFileAndIndex2
        file vpref from ref3
    output:
        file vpref into ref5
        file "${opref}.mhout" into MHOutFile
    script:
        opref = bamFile.name.replace(".bam","")
        mind = params.min_d
        msaFile = file(params.msaFile)
        dmatFile = file(params.dmatFile)
    """
    codetectmh.py -mind ${mind} -bam ${bamFile} -ref ${vpref} -msa ${msaFile} -dmat ${dmatFile}  > ${opref}.mhout
    """
}
