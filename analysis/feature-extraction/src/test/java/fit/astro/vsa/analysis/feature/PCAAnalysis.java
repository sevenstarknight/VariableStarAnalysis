package fit.astro.vsa.analysis.feature;

/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


import fit.astro.vsa.analysis.feature.PCA;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import fit.astro.vsa.common.utilities.io.MatlabFunctions;
import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;


/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class PCAAnalysis {

    public static void main(String[] args) throws IOException {

        GrabIrisData grabIrisData = new GrabIrisData();
        Map<Integer, RealVector> setOfPatterns = grabIrisData.getSetOfPatterns();

        PCA pca = new PCA(setOfPatterns);

        List<RealVector> listPCA = pca.getTransformedData(2);

        RealMatrix pcaMatrix = MatrixUtils.createRealMatrix(listPCA.size(), 2);

        int counter = 0;
        for (RealVector vector : listPCA) {
            pcaMatrix.setRowVector(counter, vector);
            counter++;
        }

        // =========================================
        MLDouble pcaArray = new MLDouble("pcaResults", pcaMatrix.getData());

        List<MLArray> list = new ArrayList<>();
        list.add(pcaArray);

        MatlabFunctions.storeToTestAnalysis("mat_PCA.mat", list);

    }
}
