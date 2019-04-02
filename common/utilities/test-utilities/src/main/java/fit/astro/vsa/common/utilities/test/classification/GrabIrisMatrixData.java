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
package fit.astro.vsa.common.utilities.test.classification;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import fit.astro.vsa.common.utilities.io.MatlabFunctions;
import java.util.HashSet;
import java.util.Set;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * https://en.wikipedia.org/wiki/Iris_flower_data_set
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class GrabIrisMatrixData {

    // =============================================
    //Output
    private final Map<Integer, RealMatrix> setOfPatterns;
    private final Map<Integer, String> setOfClasses;
    
    private final Set<String> uniqueLabels;

    /**
     * R. A. Fisher (1936). "The use of multiple measurements in taxonomic
     * problems". Annals of Eugenics. 7 (2): 179–188.
     * doi:10.1111/j.1469-1809.1936.tb02137.x.
     *
     * @throws IOException
     */
    public GrabIrisMatrixData() throws IOException {

        InputStream inStreamSignal = GrabIrisMatrixData.class
                .getResourceAsStream("/datasets/iris.mat");

        //===============================================================
        // Reading in the standard iris datasets
        MatFileReader reader = MatlabFunctions
                .generateMatFileReader(inStreamSignal);

        MLDouble measArray = (MLDouble) reader.getMLArray("meas");
        MLCell speciesArray = (MLCell) reader.getMLArray("species");

        RealMatrix measureMatrix = MatrixUtils.createRealMatrix(
                measArray.getArray());

        List<MLArray> cells = speciesArray.cells();

        // Store in object that we are expecting
        this.setOfPatterns = new HashMap<>();
        this.setOfClasses = new HashMap<>();
        this.uniqueLabels = new HashSet<>();
        for (int i = 0; i < cells.size(); i++) {
            
            RealMatrix tmpMatrix = new Array2DRowRealMatrix(2, measureMatrix.getRowVector(i).getDimension());
            
            tmpMatrix.setRowVector(0,  measureMatrix.getRowVector(i));
            tmpMatrix.setRowVector(1,  measureMatrix.getRowVector(i));
            
            setOfPatterns.put(i, tmpMatrix);
            
            
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

    }

    /**
     *
     * @return
     */
    public Map<Integer, RealMatrix> getSetOfPatterns() {
        return setOfPatterns;
    }

    /**
     *
     * @return
     */
    public Map<Integer, String> getSetOfClasses() {
        return setOfClasses;
    }

    /**
     * Get the unique classes
     * @return 
     */
    public Set<String> getUniqueLabels() {
        return uniqueLabels;
    }

    
    
}
