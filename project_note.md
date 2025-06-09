
## I. Các giai đoạn

1. **Data Import and Profiling**  
   - Import toàn bộ dữ liệu cần thiết từ các nguồn liên quan (CRM, giao dịch, thông tin sản phẩm, v.v.).
   - Thực hiện profiling nhanh để kiểm tra chất lượng dữ liệu: xác định số lượng bản ghi, kiểu dữ liệu, phân phối giá trị, phát hiện các trường có nhiều giá trị thiếu hoặc bất thường.
   - Đánh giá sơ bộ các đặc trưng quan trọng, xác định các trường có thể sử dụng cho bài toán recommendation.

2. **Data Cleaning**  
   - Xử lý missing values:
     - Nếu tỷ lệ missing < 10%, thay thế bằng mode (giá trị xuất hiện nhiều nhất) để giữ nguyên phân phối dữ liệu.
     - Nếu tỷ lệ missing > 10%, thay thế bằng mean hoặc median (nếu dữ liệu bị phân tán hoặc có outlier).
   - Loại bỏ các trường không sử dụng, các trường trùng lặp hoặc không liên quan đến bài toán.
   - Chuẩn hóa dữ liệu: chuyển đổi kiểu dữ liệu, chuẩn hóa định dạng ngày tháng, loại bỏ ký tự đặc biệt, xử lý dữ liệu ngoại lai (outlier).
   - Đảm bảo tính toàn vẹn dữ liệu, kiểm tra lại sau khi làm sạch để tránh mất mát thông tin quan trọng.

3. **Feature Engineering**  
   - Biến đổi các đặc trưng gốc thành các đặc trưng mới phù hợp cho bài toán huấn luyện:
     - Chuyển đổi các giá trị liên tục thành các nhóm (ví dụ: nhóm tuổi, nhóm thu nhập).
     - Tạo các đặc trưng tổng hợp từ nhiều trường (ví dụ: tổng số giao dịch trong 3 tháng gần nhất, số lần mua sản phẩm theo nhóm).
     - Encoding các biến phân loại (one-hot, label encoding).
   - Xem xét đặc trưng quan trọng để biến đổi, loại bỏ các đặc trưng dư thừa hoặc gây nhiễu.
   - Phân tích tương quan giữa các đặc trưng để giảm đa cộng tuyến, tăng hiệu quả mô hình.

4. **Model Training**  
   - Xây dựng và huấn luyện các mô hình được liệt kê bên dưới với dữ liệu đã chuẩn bị.
   - Chia dữ liệu thành tập train/test (hoặc validation) để đánh giá hiệu quả mô hình.
   - Tuning hyperparameters, sử dụng cross-validation nếu cần thiết.
   - Lưu lại các mô hình tốt nhất và các tham số đã thử nghiệm.

5. **Recommendation**  
   - Đề xuất sản phẩm cho khách hàng dựa trên lịch sử mua hàng, đặc trưng cá nhân và kết quả dự đoán của mô hình.
   - Đánh giá hiệu quả mô hình trên tập kiểm tra bằng các chỉ số như Precision@K, Recall@K, MAP, NDCG.
   - Phân tích các trường hợp mô hình dự đoán sai để tìm hiểu nguyên nhân và cải thiện.

6. **Implement & Improvement**  
   - Đề xuất cách ứng dụng mô hình vào thực tế: tích hợp vào hệ thống CRM, xây dựng API recommendation, dashboard cho nhân viên tư vấn.
   - Nhận diện rủi ro tiềm ẩn: dữ liệu thay đổi theo thời gian, cold-start problem, bias trong dữ liệu.
   - Đề xuất ý tưởng cải tiến: cập nhật mô hình định kỳ, bổ sung dữ liệu mới, thử nghiệm các thuật toán recommendation khác.

---

## II. Các mô hình đề xuất

### 1. Popularity-based Recommendation

- **Nguyên lý:** Đề xuất sản phẩm dựa trên tần suất mua cao nhất (item frequency) trong toàn bộ tập dữ liệu.
- **Ưu điểm:**  
  - Đơn giản, dễ triển khai, không yêu cầu dữ liệu lịch sử phức tạp.
  - Phù hợp cho mọi user, đặc biệt là user mới (cold-start), giúp họ biết được sản phẩm đang hot nhất tại thời điểm hiện tại.
- **Nhược điểm:**  
  - Không cá nhân hóa, thiếu tính personalized.
  - Không xét yếu tố thời gian, dễ dẫn đến sai lệch nếu sản phẩm chỉ hot ở một giai đoạn.
- **Giải pháp:**  
  - Áp dụng kỹ thuật *time decay adjustment*: mỗi lần tương tác với sản phẩm sẽ được gán trọng số giảm dần theo thời gian, các tương tác gần hiện tại sẽ có trọng số lớn hơn. Điều này giúp mô hình phản ánh đúng xu hướng hiện tại hơn là chỉ dựa vào tổng số lượt mua trong quá khứ.
- **Ứng dụng thực tế:**  
  - Thường dùng để khởi tạo hệ thống recommendation, hoặc làm baseline để so sánh với các mô hình phức tạp hơn.

---

### 2. Memory-based Collaborative Filtering

#### 2.1. User-based Collaborative Filtering

- **Nguyên lý:** Đề xuất cá nhân hóa dựa trên độ tương đồng giữa các user. Ý tưởng là những người dùng có sở thích giống nhau trong quá khứ sẽ có xu hướng giống nhau trong tương lai.
- **Cách hoạt động:**  
  - Tạo ma trận user-item (M x N), với M là số user, N là số sản phẩm.
  - Tính toán độ tương đồng giữa các user bằng các kỹ thuật như cosine similarity, Pearson correlation, Jaccard coefficient.
  - Xác định các user tương tự với user mục tiêu, sau đó đề xuất các sản phẩm mà họ đã mua nhưng user mục tiêu chưa mua.
- **Ưu điểm:**  
  - Cá nhân hóa tốt, tận dụng được hành vi cộng đồng.
  - Hiệu quả khi có tập người dùng lớn và nhiều dữ liệu tương tác.
- **Nhược điểm:**  
  - Ma trận user-item thường rất thưa (sparsity), nhiều giá trị 0 do phần lớn user chưa từng mua/tương tác với nhiều sản phẩm.
  - Dự đoán điểm số/gợi ý trở nên khó khăn khi dữ liệu thưa, đặc biệt với user mới hoặc sản phẩm mới (cold-start).
- **Ứng dụng thực tế:**  
  - Phù hợp với hệ thống có lượng user lớn, sản phẩm đa dạng và dữ liệu tương tác phong phú.

#### 2.2. Item-based Collaborative Filtering

- **Nguyên lý:** Đề xuất dựa trên độ tương đồng giữa các sản phẩm, thay vì giữa các user.
- **Cách hoạt động:**  
  - Tính toán độ tương đồng giữa các sản phẩm dựa trên lịch sử tương tác của user.
  - Đề xuất các sản phẩm tương tự với những sản phẩm mà user đã từng mua.
- **Ưu điểm:**  
  - Tin cậy và cá nhân hóa hơn so với user-based, vì dựa trên các sản phẩm tương tự mà khách hàng đã từng mua.
  - Thường hoạt động tốt hơn với dữ liệu thưa, vì sản phẩm thường nhận được nhiều đánh giá hơn user.
  - Được sử dụng phổ biến hơn trong công nghiệp do khả năng mở rộng và hiệu quả cao.
- **Nhược điểm:**  
  - Vẫn gặp vấn đề cold-start với sản phẩm mới chưa có lịch sử tương tác.
- **Ứng dụng thực tế:**  
  - Được sử dụng rộng rãi trong các hệ thống thương mại điện tử, ngân hàng, giải trí.

---

### 3. Model-based Collaborative Filtering

- **Nguyên lý:** Sử dụng các thuật toán machine learning để học các pattern từ dữ liệu user-item interaction, thay vì chỉ dựa vào tính toán tương đồng trực tiếp.
- **Ưu điểm so với memory-based:**
  1. **Giải quyết vấn đề dữ liệu thưa:**  
     - Sử dụng các kỹ thuật như matrix factorization, dimensionality reduction, latent factor models để phát hiện các yếu tố tiềm ẩn (latent factors) ảnh hưởng đến hành vi mua hàng.
     - Giúp khắc phục tình trạng nhiều giá trị 0 trong ma trận user-item.
  2. **Khả năng mở rộng tốt:**  
     - Mô hình có thể huấn luyện offline với các thuật toán tối ưu hóa hiệu quả, dự đoán nhanh hơn khi triển khai thực tế.
  3. **Hiệu suất cải thiện rõ rệt:**  
     - Dễ dàng tích hợp thêm các đặc trưng mới, kết hợp với các thuật toán khác để tăng hiệu quả.
- **Thuật toán sử dụng:**  
  - **Matrix Factorization:**  
    - Phân rã ma trận user-item thành hai ma trận cấp thấp hơn (user matrix & item matrix), đại diện cho mối quan hệ giữa user/item và các latent factors.
    - Ý tưởng là có các yếu tố tiềm ẩn (sở thích, đặc trưng sản phẩm) ảnh hưởng đến tương tác giữa user và item.
    - Tối ưu hóa bằng gradient descent hoặc alternating least squares để giảm sai số tái tạo giữa ma trận gốc và tích của hai ma trận mới.
    - Giúp giảm chi phí tính toán, tăng khả năng khái quát hóa và khắc phục tính thưa của dữ liệu.
- **Ứng dụng thực tế:**  
  - Được sử dụng trong các hệ thống recommendation lớn như Netflix, Amazon, các ngân hàng lớn.

---

### 4. Gradient Boosting Tree + Logistic Regression (GBDT + LR)

- **Nguyên lý:** Kết hợp ưu điểm của GBDT (khả năng học đặc trưng mạnh mẽ) và Logistic Regression (dễ hiểu, linh hoạt).
- **Cách hoạt động:**  
  - GBDT giúp nắm bắt các mẫu phức tạp và mối quan hệ tương tác giữa các đặc trưng, tự động tạo ra các đặc trưng phi tuyến tính.
  - Logistic Regression giúp giải thích mô hình và xử lý hiệu quả các đặc trưng đầu vào, đặc biệt phù hợp với dữ liệu lớn và nhiều chiều.
  - Sau khi huấn luyện, mô hình dự đoán xác suất khách hàng ưa thích từng sản phẩm dựa trên hồ sơ khách hàng, đặc trưng sản phẩm và tương tác user-item.
  - Đề xuất các sản phẩm có xác suất cao nhất cho từng khách hàng.
- **Ưu điểm:**  
  - Hiệu suất đề xuất cao, tận dụng được sức mạnh của cả hai mô hình.
  - Dễ dàng mở rộng, giải thích kết quả rõ ràng.
- **Nhược điểm:**  
  - Cần nhiều tài nguyên tính toán hơn so với các mô hình đơn giản.
  - Cần tuning hyperparameters cẩn thận để tránh overfitting.
- **Ứng dụng thực tế:**  
  - Được sử dụng trong các hệ thống recommendation hiện đại, đặc biệt khi cần giải thích kết quả cho người dùng cuối hoặc quản lý.

---

## III. Tổng kết

- Dự án triển khai qua các bước: nhập dữ liệu, làm sạch, tạo đặc trưng, huấn luyện mô hình, đánh giá và cải tiến.
- Mỗi bước đều đóng vai trò quan trọng trong việc đảm bảo chất lượng và hiệu quả của hệ thống recommendation.
- Các mô hình được đề xuất từ đơn giản (popularity-based) đến phức tạp (model-based, hybrid), mỗi mô hình có ưu nhược điểm riêng và phù hợp với từng giai đoạn phát triển của hệ thống.
- Để tối ưu hiệu quả, nên kết hợp nhiều phương pháp, liên tục cập nhật dữ liệu và cải tiến mô hình dựa trên phản hồi thực tế.
- Đặc biệt, cần chú ý đến các vấn đề thực tế như cold-start, dữ liệu thay đổi theo thời gian, và khả năng mở rộng khi triển khai trong môi trường ngân hàng.
