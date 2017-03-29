#define NAME_LEN_MAX 512
#define CARD_NUM_LEN_MAX 16

struct customer {
    char first_Name[NAME_LEN_MAX + 1];
    char last_Name[NAME_LEN_MAX + 1];
    float amount;
    char card_Number[CARD_NUM_LEN_MAX + 1];
    int card_Expiration_Month;
    int card_Expiration_Year;
    char sales_Person[NAME_LEN_MAX + 1];
};

void menu(struct customer *cp);
void cls(void);
int get_ID_Input(int a);
void menu_update(void);
