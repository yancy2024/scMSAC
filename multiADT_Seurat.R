library(ggplot2)
library(Seurat)
library(rhdf5)
library(dplyr)
library(aricode)
setwd("D:/Multi-omics/RNA_ADT/Mimitou")
# setwd("C:/Users/yancy/Desktop/sample/sample70")

###csv
cbmc.rna <- as.sparse(read.csv(file = "./RNA.csv", sep = ",",
                               header = TRUE, row.names = 1))

cbmc.rna <- CollapseSpeciesExpressionMatrix(cbmc.rna)

# Load in the ADT UMI matrix
cbmc.adt <- as.sparse(read.csv(file = "./ADT.csv", sep = ",",
                               header = TRUE, row.names = 1))

#all.equal(colnames(cbmc.rna), colnames(cbmc.adt))
label<-read.table("labels.csv",header=T,row.names=1,sep=",",check.names=F)
label<-as.matrix(label)


###h5, choose one
dat <- H5Fopen("./CITESeq_realdata_spleen_lymph_111_anno_multiBatch.h5")

#X is count matrix; Y is the true label (used for remove NA cells); Pos is the spatial coordinates.
label <- dat$Y
cbmc.rna <- as.data.frame(dat$X1)
cbmc.adt <- as.data.frame(dat$X2)


row_names <- paste("Row", 1:nrow(cbmc.rna))
col_names <- paste("Column", 1:ncol(cbmc.rna))
row_names1 <- paste("Row", 1:nrow(cbmc.adt))
col_names1 <- paste("Column", 1:ncol(cbmc.adt))


for (i in 1:nrow(cbmc.rna)) {
  rownames(cbmc.rna)[i] <- row_names[i]
}

for (i in 1:nrow(cbmc.adt)) {
  rownames(cbmc.adt)[i] <- row_names1[i]
}

for (j in 1:ncol(cbmc.rna)) {
  colnames(cbmc.rna)[j] <- col_names[j]
}

for (j in 1:ncol(cbmc.adt)) {
  colnames(cbmc.adt)[j] <- col_names1[j]
}




# creates a Seurat object based on the scRNA-seq data
cbmc <- CreateSeuratObject(counts = cbmc.rna)
# create a new assay to store ADT information
adt_assay <- CreateAssayObject(counts = cbmc.adt)
# add this assay to the previously created Seurat object
cbmc[["ADT"]] <- adt_assay


# default assay is RNA
DefaultAssay(cbmc) <- "RNA"
cbmc <- cbmc %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()

# Normalize ADT data
DefaultAssay(cbmc) <- "ADT"
VariableFeatures(cbmc) <- rownames(cbmc[["ADT"]])
cbmc <- cbmc %>% ScaleData() %>% RunPCA(reduction.name = 'apca')

#head(cbmc@reductions$apca@cell.embeddings)

cbmc <- FindMultiModalNeighbors(
  cbmc, reduction.list = list("pca", "apca"), 
  dims.list = list(1:30, 1:10), modality.weight.name = "RNA.weight"
)

#save WNN graph
write.table(cbmc[["wsnn"]], 'wnn_adj.csv', quote=FALSE, sep=',',col.names=T,row.names=T)
#

cbmc <- RunUMAP(cbmc, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")

# merged_reduction <- cbind(cbmc@reductions$pca@cell.embeddings, cbmc@reductions$apca@cell.embeddings)
# write.table(merged_reduction, 'seurat_embedding.csv', quote=FALSE, sep=',',col.names=T,row.names=T)

bmc <- FindClusters(cbmc, graph.name = "wsnn", algorithm = 3, resolution =0.5, verbose = FALSE)
pre<-Idents(bmc)
la <- as.vector(label)
# la <- label[,2]
ari <- ARI(la, pre)
nmi <- NMI(la, pre)

write.table(pre, 'seu_pre.csv', quote=FALSE, sep=',',col.names=T,row.names=T)














