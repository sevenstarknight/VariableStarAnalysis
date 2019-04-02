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
package fit.astro.vsa.common.utilities.io.dataset;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;
import fit.astro.vsa.common.bindings.analysis.astro.VarStarDataset;
import fit.astro.vsa.common.bindings.analysis.astro.VarStarInformation;
import fit.astro.vsa.common.bindings.math.ml.input.MultiView;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.io.MatlabFunctions;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.math3.analysis.function.Inverse;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class ReadingInLINEARData {

    public enum Views {
        Photometric, TimeDomainVar, SSMM, DF, Statistics;
    }

    private final VarStarDataset linearDataset;

    public ReadingInLINEARData(String fileLocation) throws IOException {

        File inputFile = new File(fileLocation);

        InputStream finPCA = new FileInputStream(inputFile);

        MatFileReader reader = MatlabFunctions
                .generateMatFileReader(finPCA);

        // ==========================
        MLStructure structMC = (MLStructure) reader.getMLArray("structMCReduct");
        MLCell grpSource = (MLCell) reader.getMLArray("grpSource");

        Map<Integer, String> setOfClasses = new HashMap<>();

        // ==========================
        Map<Integer, VarStarInformation> info = new HashMap<>();

        // ==========================
        Map<Integer, Map<String, RealVector>> patterns = new HashMap<>();
        Map<Integer, Real2DCurve> timeD = new HashMap<>();

        for (int idx = 0; idx < grpSource.getSize(); idx++) {

            MLChar labelCh = (MLChar) grpSource.get(idx);
            String label = labelCh.contentToString();
            setOfClasses.put(idx, label);

            MLChar id = (MLChar) structMC.getField("ID", idx);
            VarStarInformation information = new VarStarInformation(id.contentToString());
            info.put(idx, information);

            // ug, gi, iK, JK, logP, Ampl, skew, kurt, magMed
            MLDouble parameters
                    = (MLDouble) structMC.getField("parameters", idx);

            RealVector paraMatrix = MatrixUtils.createRealMatrix(
                    parameters.getArray()).getRowVector(0);

            patterns.put(idx, new HashMap<>());

            // index location known based on expert knowledge
            patterns.get(idx).put(Views.Photometric.toString(), paraMatrix.getSubVector(0, 4));

            patterns.get(idx).put(Views.TimeDomainVar.toString(), paraMatrix.getSubVector(4, 5));

            MLDouble timeDomainArray
                    = (MLDouble) structMC.getField("timeSeries", idx);

            RealMatrix tdMatrix = MatrixUtils.createRealMatrix(
                    timeDomainArray.getArray());

            // ==========================
            Real2DCurve twoDSeries = new Real2DCurve(
                    tdMatrix.getColumnVector(0), tdMatrix.getColumnVector(1),
                    tdMatrix.getColumnVector(2).map(new Inverse()));

            timeD.put(idx, twoDSeries);

        }

        
        MultiView multiViewSideData = new MultiView(
                "Variable Stars", patterns, new HashMap<>(), setOfClasses);

        this.linearDataset = new VarStarDataset("LINEAR", timeD, info);
        linearDataset.setMultiViewSideData(multiViewSideData);

    }

    private void generateWhiteNoise(){
        
        
        
        
    }
    
    
    public VarStarDataset getLinearDataset() {
        return linearDataset;
    }

}
