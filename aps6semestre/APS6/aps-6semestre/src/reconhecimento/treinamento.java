/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reconhecimento;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;
import javax.swing.JOptionPane;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face.EigenFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_face.FisherFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.LBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 *
 * @author gugal
 */
public class treinamento {
    public static void main(String args[]) {
        
    }
    public static void treinar (){
        File diretorio = new File("src\\fotos");
        FilenameFilter filtroImagem = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String nome) {
                return nome.endsWith(".jpg") || nome.endsWith(".gif") || nome.endsWith(".png");
            }
        };
        File[] arquivos = diretorio.listFiles(filtroImagem);
        MatVector fotos = new opencv_core.MatVector(arquivos.length);
        Mat rotulos = new opencv_core.Mat(arquivos.length, 1, CV_32SC1);
        IntBuffer rotulosBuffer = rotulos.createBuffer();
        int contador = 0;
        for (File imagem: arquivos) {
            Mat foto = imread(imagem.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int classe = Integer.parseInt(imagem.getName().split("\\.")[1]);
            System.out.println(classe);
            resize(foto, foto, new Size(160,160));
            fotos.put(contador, foto);
            rotulosBuffer.put(contador, classe);
            contador++;
        }
        
        FaceRecognizer eigenfaces = EigenFaceRecognizer.create(25, 0);
        FaceRecognizer fisherfaces = FisherFaceRecognizer.create();
        FaceRecognizer lbph = LBPHFaceRecognizer.create(2,9,9,9,1);
        
        eigenfaces.train(fotos, rotulos);
        eigenfaces.save("src\\recursos\\classificadorEigenFaces.yml");
        fisherfaces.train(fotos, rotulos);
        fisherfaces.save("src\\recursos\\classificadorFisherFaces.yml");
        lbph.train(fotos, rotulos);
        lbph.save("src\\recursos\\classificadorLBPH.yml");
        
        JOptionPane.showMessageDialog (null, "Treinamento realizado, já pode reconhecer o rosto");
    }
}
