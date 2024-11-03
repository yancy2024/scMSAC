library(Seurat)
library(Signac)
library(dplyr)
library(ggplot2)
library(rhdf5)
# library(EnsDb.Mmusculus.v79)
library(EnsDb.Hsapiens.v86)
library(aricode)
library(SeuratData)
setwd("D:/Multi-omics/RNA_ATAC/pbmc10k/")

pbmc <- LoadData("pbmcMultiome", "pbmc.rna")
atac_counts <- LoadData("pbmcMultiome", "pbmc.atac")
pbmc <- subset(pbmc, seurat_annotations != "filtered")
atac_counts <- subset(atac_counts, seurat_annotations != "filtered")

label<-read.table("./labels.csv",header=T,row.names=1,sep=",",check.names=F)
label<-as.matrix(label)
# 
# # extract RNA and ATAC data
rna_counts <- as.sparse(read.csv(file = "./RNA.csv", sep = ",",
                               header = TRUE, row.names = 1))

atac_counts <- as.sparse(read.csv(file = "./ATAC.csv", sep = ",",
                                 header = TRUE, row.names = 1))

# # 读取H5AD文件
# library(SCP)
# library(reticulate)
# sc <- import("scanpy")
# rna <- sc$read_h5ad("Chen-2019-RNA.h5ad")
# atac <- sc$read_h5ad("Chen-2019-ATAC.h5ad")
# # write.table(atac$`_obs`[["cell_type"]], 'label.csv', quote=FALSE, sep=',',col.names=T,row.names=T)
# 
# pbmc <- adata_to_srt(rna)
# atac_counts <- adata_to_srt(atac)


###h5, choose one
dat <- H5Fopen("D:/Multi-omics/RNA_ATAC/SMAGESeq_10X_pbmc_granulocyte_plus.h5")

#X is count matrix; Y is the true label (used for remove NA cells); Pos is the spatial coordinates.
label <- dat$Y
cbmc.rna <- as.data.frame(dat$X1)
cbmc.atac <- as.data.frame(dat$X2)


row_names <- paste("Row", 1:nrow(cbmc.rna))
col_names <- paste("Column", 1:ncol(cbmc.rna))
row_names1 <- paste("Row", 1:nrow(cbmc.atac))
col_names1 <- paste("Column", 1:ncol(cbmc.atac))


for (i in 1:nrow(cbmc.rna)) {
  rownames(cbmc.rna)[i] <- row_names[i]
}

for (i in 1:nrow(cbmc.atac)) {
  rownames(cbmc.atac)[i] <- row_names1[i]
}

for (j in 1:ncol(cbmc.rna)) {
  colnames(cbmc.rna)[j] <- col_names[j]
}

for (j in 1:ncol(cbmc.atac)) {
  colnames(cbmc.atac)[j] <- col_names1[j]
}





# Create Seurat object
pbmc <- CreateSeuratObject(counts = rna_counts)
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")

# Now add in the ATAC-seq data
# we'll only use peaks in standard chromosomes
grange.counts <- StringToGRanges(rownames(atac_counts), sep = c(":", "-"))
grange.use <- seqnames(grange.counts) %in% standardChromosomes(grange.counts)
atac_counts <- atac_counts[as.vector(grange.use), ]

chrom_assay <- CreateChromatinAssay(
  counts = atac_counts,      #@assays$ATAC@counts
  sep = c(":", "-"),
  min.cells = 10
)
pbmc[["ATAC"]] <- chrom_assay



# RNA analysis
DefaultAssay(pbmc) <- "RNA"
pbmc <- SCTransform(pbmc, verbose = FALSE) %>% RunPCA() %>% RunUMAP(dims = 1:50, reduction.name = 'umap.rna', reduction.key = 'rnaUMAP_')

# ATAC analysis
# We exclude the first dimension as this is typically correlated with sequencing depth
DefaultAssay(pbmc) <- "ATAC"
pbmc <- RunTFIDF(pbmc)
pbmc <- FindTopFeatures(pbmc, min.cutoff = 'q0')
pbmc <- RunSVD(pbmc)
pbmc <- RunUMAP(pbmc, reduction = 'lsi', dims = 2:50, reduction.name = "umap.atac", reduction.key = "atacUMAP_")

# merged_reduction <- cbind(pbmc@reductions$pca@cell.embeddings, pbmc@reductions$lsi@cell.embeddings)
# save lsi_features
# lsi_embedding <- t(pbmc@reductions$lsi@cell.embeddings)
# write.table(lsi_embedding, 'ATAC_lsi.csv', quote=FALSE, sep=',',col.names=T,row.names=T)

pbmc <- FindMultiModalNeighbors(pbmc, reduction.list = list("pca", "lsi"), dims.list = list(1:50, 2:50))

#保存WNN图，可选
write.table(pbmc[["wsnn"]], 'wnn_adj.csv', quote=FALSE, sep=',',col.names=T,row.names=T)


pbmc <- RunUMAP(pbmc, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
bmc <- FindClusters(pbmc, graph.name = "wsnn", algorithm = 3, resolution =3, verbose = FALSE)
pre<-Idents(bmc)
la <- as.vector(label)
ari <- ARI(la, pre)
nmi <- NMI(la, pre)

write.table(pre, 'seu_pre.csv', quote=FALSE, sep=',',col.names=T,row.names=T)






