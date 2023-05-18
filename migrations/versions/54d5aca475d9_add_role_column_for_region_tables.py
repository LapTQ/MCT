"""add role column for region tables

Revision ID: 54d5aca475d9
Revises: b4187f95bad9
Create Date: 2023-05-18 17:45:42.808879

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '54d5aca475d9'
down_revision = 'b4187f95bad9'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('region', schema=None) as batch_op:
        batch_op.add_column(sa.Column('role', sa.String(length=128), nullable=True))
        batch_op.create_foreign_key('fk_region_user_role', 'user', ['role'], ['role'])

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('region', schema=None) as batch_op:
        batch_op.drop_constraint('fk_region_user_role', type_='foreignkey')
        batch_op.drop_column('role')

    # ### end Alembic commands ###
