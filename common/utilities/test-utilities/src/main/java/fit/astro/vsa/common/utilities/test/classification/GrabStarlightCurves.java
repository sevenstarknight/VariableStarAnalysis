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
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import fit.astro.vsa.common.utilities.io.MatlabFunctions;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import java.util.HashSet;
import java.util.Set;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class GrabStarlightCurves {

    // =============================================
    //Output
    private final Map<Integer, RealVector> setOfPatterns;
    private final Map<Integer, String> setOfClasses;

    private final Set<String> uniqueLabels;
    
    /**
     * Constructor with no arguments
     * <p>
     * Pavlos Protopapas, JM Giammarco, L Faccioli, MF Struble, Rahul Dave, and
     * Charles Alcock. Finding outlier light curves in catalogues of periodic
     * variable stars. Monthly Notices of the Royal Astronomical Society,
     * 369(2): 677–696, 2006
     *
     * @throws IOException
     */
    public GrabStarlightCurves() throws IOException {

        //===============================================================
        InputStream inStreamSignal = GrabStarlightCurves.class
                .getResourceAsStream("/datasets/UCRStarlight.mat");

        MatFileReader reader = MatlabFunctions.generateMatFileReader(inStreamSignal);

        //===============================================================
        MLDouble curveArray;
        Object curveObj = reader.getMLArray("fltDataSet");
        if (curveObj instanceof MLDouble) {
            curveArray = (MLDouble) curveObj;
        } else {
            throw new IOException("fltDataSet was not of type MLDouble!");
        }

        MLCell labelArray;
        Object labelObj = reader.getMLArray("grpSource");
        if (labelObj instanceof MLCell) {
            labelArray = (MLCell) labelObj;
        } else {
            throw new IOException("grpSource was not of type MLCell!");
        }

        //===============================================================
        RealMatrix measureMatrix = MatrixUtils.createRealMatrix(curveArray.getArray());

        //===============================================================
        List<MLArray> cells = labelArray.cells();

        this.setOfPatterns = new HashMap<>();
        this.setOfClasses = new HashMap<>();
        this.uniqueLabels = new HashSet<>();

        for (int i = 0; i < cells.size(); i++) {
            setOfPatterns.put(i, measureMatrix.getRowVector(i));
            MLChar nameTemp = (MLChar) cells.get(i);
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
     * RealVector times = MatrixUtils.createRealVector(VectorOperations.linearSpace(
 0.05 * (double) amplitudes.getDimension(), 0.0,
 amplitudes.getDimension()));
     *
     * @return tack on times (because the real Starlight curves dont have times
     *
     */
    public Map<Integer, Real2DCurve> getSetOfWaveforms() {

        Map<Integer, Real2DCurve> mapOfWaveforms = new HashMap<>();

        for (Integer idx : setOfPatterns.keySet()) {

            RealVector amplitudes = setOfPatterns.get(idx);

            RealVector times = MatrixUtils.createRealVector(VectorOperations.linearSpace(
                    0.05 * (double) amplitudes.getDimension(), 0.0,
                    amplitudes.getDimension()));

            mapOfWaveforms.put(idx, new Real2DCurve(times, amplitudes));
        }
        return mapOfWaveforms;
    }

    /**
     * Get the set of patterns.
     *
     * @return
     */
    public Map<Integer, RealVector> getSetOfPatterns() {
        return setOfPatterns;
    }

    /**
     * Set the classes map.
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
