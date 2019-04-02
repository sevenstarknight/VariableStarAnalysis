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
 * but WITHOUT ANY WARRANTY; without isEven the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


import fit.astro.vsa.analysis.feature.PCA;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;
import fit.astro.vsa.common.utilities.io.MatlabFunctions;
import fit.astro.vsa.common.utilities.math.NumericTests;
import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import java.io.IOException;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import java.io.InputStream;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class PCATestNew {

    public PCATestNew() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    @Test
    public void testPC() throws IOException {

        GrabIrisData grabIrisData = new GrabIrisData();
        Map<Integer, RealVector> setOfPatterns
                = grabIrisData.getSetOfPatterns();

        PCA pca = new PCA(setOfPatterns);

        List<RealVector> listPCA = pca.getTransformedData(2);

        RealMatrix pcaMatrix = MatrixUtils.createRealMatrix(
                listPCA.size(), 2);

        int counter = 0;
        for (RealVector vector : listPCA) {
            pcaMatrix.setRowVector(counter, vector);
            counter++;
        }

        InputStream finPCA = PCATestNew.class
                .getResourceAsStream("/analysis/mat_PCA.mat");
        MatFileReader reader = MatlabFunctions
                .generateMatFileReader(finPCA);

        MLDouble inputData = (MLDouble) reader
                .getMLArray("pcaResults");

        RealMatrix inputMatrix = MatrixUtils
                .createRealMatrix(inputData.getArray());

        double error = pcaMatrix.subtract(inputMatrix)
                .getFrobeniusNorm();

        assertEquals(Boolean.TRUE, NumericTests.isApproxZero(error));

    }
}
