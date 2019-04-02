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
package fit.astro.vsa.utilities.ml.ecva;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import fit.astro.vsa.common.utilities.io.MatlabFunctions;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author SevenStarKnight
 */
public class ECVAAnalysis {

    private static Map<Integer, RealVector> setOfPatterns;
    private static Map<Integer, String> setOfClasses;
    
    private static Set<String> uniqueLabels;

    public static void main(String[] args) throws IOException {

        InputStream inStreamSignal = ECVAAnalysis.class
                .getResourceAsStream("/data/ucrDataStarlightSSMM.mat");

        //===============================================================
        // Reading in the standard iris datasets
        MatFileReader reader = MatlabFunctions
                .generateMatFileReader(inStreamSignal);

        MLDouble measArray = (MLDouble) reader.getMLArray("spaceTotal");
        MLCell speciesArray = (MLCell) reader.getMLArray("grpSource");

        RealMatrix measureMatrix = MatrixUtils.createRealMatrix(
                measArray.getArray());

        List<MLArray> cells = speciesArray.cells();

        // Store in object that we are expecting
        setOfPatterns = new HashMap<>();
        setOfClasses = new HashMap<>();
        uniqueLabels = new HashSet<>();
        for (int i = 0; i < cells.size(); i++) {
            setOfPatterns.put(i,
                    measureMatrix.getRowVector(i));
            MLChar nameTemp = (MLChar) cells.get(i);
            // Parse strings
            Character[] arrayChar = nameTemp.exportChar();

            StringBuilder sb = new StringBuilder();

            for (Character c : arrayChar) {
                sb.append(c.toString());
            }

            uniqueLabels.add(sb.toString());
            setOfClasses.put(i, sb.toString());
        }

        // =========================================
        ECVA ecva = new ECVA(setOfPatterns, setOfClasses);

        CanonicalVariates canonicalVariates = ecva.execute();

        // =========================================
        RealMatrix cvaMatrix = new Array2DRowRealMatrix(
                canonicalVariates.getCanonicalVariates().size(), 2);
        int counter = 0;
        for (Integer idx : canonicalVariates.getCanonicalVariates().keySet()) {

            RealVector cva = canonicalVariates.getCanonicalVariates().get(idx);

            cvaMatrix.setRowVector(counter, cva);
            counter++;
        }

        MLArray errorML = new MLDouble("CVA", cvaMatrix.getData());
        List<MLArray> list = new ArrayList<>();
        list.add(errorML);

        MatlabFunctions.storeToTestAnalysis("CVAData.mat", list);

    }
}
